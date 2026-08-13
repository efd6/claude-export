package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// discoverProjects returns all project directories under root.
func discoverProjects(root string) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("reading projects directory: %w", err)
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, filepath.Join(root, e.Name()))
		}
	}
	sort.Strings(dirs)
	return dirs, nil
}

// projectCWD returns the working directory stored in the first session
// record of the first JSONL file in dir. Returns empty string on failure.
func projectCWD(dir string) (string, error) {
	files, err := sessionFiles(dir)
	if err != nil || len(files) == 0 {
		return "", nil
	}
	f, err := os.Open(files[0])
	if err != nil {
		return "", err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 16<<20), 16<<20)
	var cwd string
	for sc.Scan() {
		var r struct {
			CWD string `json:"cwd"`
		}
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.CWD != "" {
			cwd = r.CWD
			break
		}
	}
	return cwd, sc.Err()
}

// sessionFiles returns all JSONL files directly in dir (not in subdirs).
func sessionFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading directory: %w", err)
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(files)
	return files, nil
}

// rawRecord is a partially decoded JSONL record.
type rawRecord struct {
	Type            string        `json:"type"`
	Subtype         string        `json:"subtype"`
	UUID            string        `json:"uuid"`
	ParentUUID      *string       `json:"parentUuid"`
	IsSidechain     bool          `json:"isSidechain"`
	Timestamp       string        `json:"timestamp"`
	CWD             string        `json:"cwd"`
	AITitle         string        `json:"aiTitle"`
	LeafUUID        string        `json:"leafUuid"`
	LastPrompt      string        `json:"lastPrompt"`
	Message         *rawMessage   `json:"message"`
	CompactMetadata *compactMeta  `json:"compactMetadata"`
}

// compactMeta holds the subset of compactMetadata we need to cross
// compaction boundaries during chain traversal.
type compactMeta struct {
	PreservedSegment struct {
		TailUUID string `json:"tailUuid"`
	} `json:"preservedSegment"`
}

// rawMessage is the partially decoded message field within a rawRecord.
type rawMessage struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Content json.RawMessage `json:"content"`
}

// contentBlock is one item in a message content array.
type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"` // string or block array for tool_result
}

// parseSession reads a JSONL session file and returns the reconstructed
// session and the working directory recorded in the session records.
func parseSession(path, id string, extra bool) (session, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return session{}, "", fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	var records []rawRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 16<<20), 16<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r rawRecord
		if json.Unmarshal(line, &r) == nil {
			records = append(records, r)
		}
	}
	if err := sc.Err(); err != nil {
		return session{}, "", fmt.Errorf("reading: %w", err)
	}
	if len(records) == 0 {
		return session{}, "", fmt.Errorf("empty file")
	}

	// Use the last ai-title record as the session title.
	var title string
	for _, r := range records {
		if r.Type == "ai-title" && r.AITitle != "" {
			title = r.AITitle
		}
	}

	// Find the leaf UUID from the last last-prompt record that has a
	// lastPrompt value; fall back to the first last-prompt otherwise.
	var leafUUID string
	for _, r := range records {
		if r.Type == "last-prompt" {
			if r.LastPrompt != "" {
				leafUUID = r.LeafUUID
			} else if leafUUID == "" {
				leafUUID = r.LeafUUID
			}
		}
	}
	if leafUUID == "" {
		return session{}, "", fmt.Errorf("no leaf UUID")
	}

	// Build a UUID → record index.
	byUUID := make(map[string]*rawRecord, len(records))
	for i := range records {
		if records[i].UUID != "" {
			byUUID[records[i].UUID] = &records[i]
		}
	}

	// Walk the parentUuid chain from leaf to root, then reverse to get
	// chronological order. compact_boundary records are not added to the
	// chain; instead, traversal continues via preservedSegment.tailUuid so
	// pre-compaction messages are included transparently.
	var chain []*rawRecord
	seen := make(map[string]struct{})
	for cur := leafUUID; cur != ""; {
		if _, ok := seen[cur]; ok {
			break
		}
		r, ok := byUUID[cur]
		if !ok {
			break
		}
		seen[cur] = struct{}{}
		if r.Type == "system" && r.Subtype == "compact_boundary" {
			if r.CompactMetadata != nil && r.CompactMetadata.PreservedSegment.TailUUID != "" {
				cur = r.CompactMetadata.PreservedSegment.TailUUID
				continue
			}
			break
		}
		chain = append(chain, r)
		if r.ParentUUID == nil {
			break
		}
		cur = *r.ParentUUID
	}
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}

	// Extract the working directory from the first record that has one.
	var cwd string
	for _, r := range chain {
		if r.CWD != "" {
			cwd = r.CWD
			break
		}
	}

	// Find the session creation time from the first user record.
	var createdAt time.Time
	for _, r := range chain {
		if r.Type == "user" {
			t, _ := time.Parse(time.RFC3339Nano, r.Timestamp)
			if !t.IsZero() {
				createdAt = t
				break
			}
		}
	}

	msgs := buildMessages(chain, extra)

	return session{
		ID:        id,
		Title:     title,
		CreatedAt: createdAt,
		Messages:  msgs,
	}, cwd, nil
}

// buildMessages converts a chain of raw records into conversation messages.
// Consecutive assistant records with the same API message ID are combined
// into a single logical turn.
func buildMessages(chain []*rawRecord, extra bool) []message {
	var msgs []message

	var (
		pendingID      string
		pendingModel   string
		pendingTS      time.Time
		pendingTexts   []string
		pendingThought string
		pendingTools   []*toolUse
	)

	flushPending := func() {
		if pendingID == "" {
			return
		}
		msg := message{
			Role:      "assistant",
			Timestamp: pendingTS,
			Model:     pendingModel,
		}
		if extra && pendingThought != "" {
			msg.Thinking = pendingThought
		}
		if len(pendingTexts) > 0 {
			msg.Text = strings.Join(pendingTexts, "\n")
		}
		if msg.Text != "" || (extra && msg.Thinking != "") {
			msgs = append(msgs, msg)
		}
		if extra {
			for _, tu := range pendingTools {
				msgs = append(msgs, message{
					Role:      "tool",
					Timestamp: pendingTS,
					ToolUse:   tu,
				})
			}
		}
		pendingID = ""
		pendingModel = ""
		pendingTS = time.Time{}
		pendingTexts = nil
		pendingThought = ""
		pendingTools = nil
	}

	for _, r := range chain {
		switch r.Type {
		case "user":
			flushPending()
			if r.Message == nil {
				continue
			}
			ts, _ := time.Parse(time.RFC3339Nano, r.Timestamp)
			msgs = append(msgs, extractUserContent(r.Message.Content, ts, extra)...)

		case "assistant":
			if r.Message == nil {
				continue
			}
			ts, _ := time.Parse(time.RFC3339Nano, r.Timestamp)
			if r.Message.ID != pendingID {
				flushPending()
				pendingID = r.Message.ID
				pendingModel = r.Message.Model
				pendingTS = ts
			}
			texts, thought, tools := extractAssistantContent(r.Message.Content)
			pendingTexts = append(pendingTexts, texts...)
			if thought != "" {
				pendingThought = thought
			}
			pendingTools = append(pendingTools, tools...)
		}
	}
	flushPending()
	return msgs
}

// extractUserContent parses a user message content value into messages.
func extractUserContent(raw json.RawMessage, ts time.Time, extra bool) []message {
	if len(raw) == 0 {
		return nil
	}

	// Content can be a plain string (early Claude CLI versions).
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return []message{{Role: "user", Timestamp: ts, Text: s}}
	}

	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return nil
	}

	var msgs []message
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				msgs = append(msgs, message{Role: "user", Timestamp: ts, Text: b.Text})
			}
		case "tool_result":
			if !extra {
				continue
			}
			msgs = append(msgs, message{
				Role:      "tool_result",
				Timestamp: ts,
				ToolResult: &toolResult{
					ToolUseID: b.ToolUseID,
					Content:   toolResultContent(b.Content),
				},
			})
		}
	}
	return msgs
}

// toolResultContent extracts a string from a tool_result content field,
// which may be a JSON string or an array of text blocks.
func toolResultContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	var parts []string
	for _, b := range blocks {
		if b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// extractAssistantContent parses an assistant message content array,
// returning text, the first thinking block, and tool use calls.
func extractAssistantContent(raw json.RawMessage) (texts []string, thinking string, tools []*toolUse) {
	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return
	}
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				texts = append(texts, b.Text)
			}
		case "thinking":
			if b.Thinking != "" && thinking == "" {
				thinking = b.Thinking
			}
		case "tool_use":
			tools = append(tools, &toolUse{
				ID:    b.ID,
				Name:  b.Name,
				Input: b.Input,
			})
		}
	}
	return
}
