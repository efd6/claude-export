package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
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

// sessionGroup holds the files belonging to one logical session,
// ordered from earliest to latest.
type sessionGroup struct {
	id    string
	files []string
}

// groupSessionFiles groups JSONL files by logical session, linking
// continuation files to their predecessors.
func groupSessionFiles(files []string) ([]sessionGroup, error) {
	pathByID := make(map[string]string, len(files))
	for _, f := range files {
		id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		pathByID[id] = f
	}

	pred := make(map[string]string)
	for _, f := range files {
		id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		p, err := findPredecessor(f)
		if err != nil || p == "" {
			continue
		}
		pred[id] = p
	}

	succ := make(map[string]string, len(pred))
	for child, parent := range pred {
		succ[parent] = child
	}

	grouped := make(map[string]bool, len(files))
	var groups []sessionGroup
	for _, f := range files {
		id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		if grouped[id] {
			continue
		}
		root := id
		for {
			p, ok := pred[root]
			if !ok {
				break
			}
			root = p
		}
		var chain []string
		cur := root
		for {
			if path, ok := pathByID[cur]; ok {
				chain = append(chain, path)
			}
			grouped[cur] = true
			next, ok := succ[cur]
			if !ok {
				break
			}
			cur = next
		}
		if len(chain) > 0 {
			groups = append(groups, sessionGroup{
				id:    strings.TrimSuffix(filepath.Base(chain[len(chain)-1]), ".jsonl"),
				files: chain,
			})
		}
	}

	return groups, nil
}

// findPredecessor scans a session file for records with a session_id
// field that differs from the file's own session ID, indicating this
// file is a continuation of an earlier session.
func findPredecessor(path string) (string, error) {
	fileID := strings.TrimSuffix(filepath.Base(path), ".jsonl")
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 16<<20), 16<<20)
	for sc.Scan() {
		var r struct {
			SessionID string `json:"session_id"`
		}
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.SessionID != "" && r.SessionID != fileID {
			return r.SessionID, nil
		}
	}
	return "", sc.Err()
}

// rawRecord is a partially decoded JSONL record.
type rawRecord struct {
	Type            string       `json:"type"`
	Subtype         string       `json:"subtype"`
	UUID            string       `json:"uuid"`
	ParentUUID      *string      `json:"parentUuid"`
	IsSidechain     bool         `json:"isSidechain"`
	Timestamp       string       `json:"timestamp"`
	CWD             string       `json:"cwd"`
	AITitle         string       `json:"aiTitle"`
	LeafUUID        string       `json:"leafUuid"`
	LastPrompt      string       `json:"lastPrompt"`
	Message         *rawMessage  `json:"message"`
	CompactMetadata *compactMeta `json:"compactMetadata"`
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

// parseSession reads one or more JSONL session files belonging to
// the same logical session and returns the reconstructed session and
// the working directory. When multiple files are provided they must
// be ordered from oldest to newest.
func parseSession(paths []string, id string, extra bool) (session, string, error) {
	var perFile [][]rawRecord
	for _, path := range paths {
		records, err := readRecords(path)
		if err != nil {
			return session{}, "", fmt.Errorf("reading %s: %w", path, err)
		}
		perFile = append(perFile, records)
	}

	empty := true
	for _, records := range perFile {
		if len(records) > 0 {
			empty = false
			break
		}
	}
	if empty {
		return session{}, "", fmt.Errorf("empty session")
	}

	// Use the last ai-title record across all files as the session title.
	var title string
	for _, records := range perFile {
		for _, r := range records {
			if r.Type == "ai-title" && r.AITitle != "" {
				title = r.AITitle
			}
		}
	}

	// Find the leaf UUID from the last last-prompt record that has a
	// lastPrompt value; fall back to the first last-prompt otherwise.
	var leafUUID string
	for _, records := range perFile {
		for _, r := range records {
			if r.Type == "last-prompt" {
				if r.LastPrompt != "" {
					leafUUID = r.LeafUUID
				} else if leafUUID == "" {
					leafUUID = r.LeafUUID
				}
			}
		}
	}
	if leafUUID == "" {
		return session{}, "", fmt.Errorf("no leaf UUID")
	}

	// Build UUID → record index. Within each file the last record
	// with a given UUID wins (in-file updates). Across files the
	// oldest version takes priority so that the parentUuid chain of
	// preserved records leads into the earlier file's history rather
	// than back to the compact_boundary.
	//
	// Also build a map from each compact_boundary UUID to the leaf
	// UUID of the preceding segment (from the last-prompt record
	// before the boundary). When the boundary's preserved-segment
	// tail has already been visited (cycle within one file), this
	// leaf lets the traversal continue into the previous segment.
	byUUID := make(map[string]*rawRecord)
	boundaryLeaf := make(map[string]string)
	for fi := range perFile {
		fileIdx := make(map[string]*rawRecord, len(perFile[fi]))
		var lastLeaf string
		for i := range perFile[fi] {
			r := &perFile[fi][i]
			if r.UUID != "" {
				fileIdx[r.UUID] = r
			}
			if r.Type == "last-prompt" && r.LeafUUID != "" {
				lastLeaf = r.LeafUUID
			}
			if r.Type == "system" && r.Subtype == "compact_boundary" && r.UUID != "" && lastLeaf != "" {
				boundaryLeaf[r.UUID] = lastLeaf
			}
		}
		for uuid, r := range fileIdx {
			if _, exists := byUUID[uuid]; !exists {
				byUUID[uuid] = r
			}
		}
	}

	// Walk the parentUuid chain from leaf to root, then reverse to get
	// chronological order. compact_boundary records are not added to the
	// chain; instead, traversal continues via preservedSegment.tailUuid so
	// pre-compaction messages are included transparently.
	//
	// When a boundary's tail has already been visited the preserved
	// records form a cycle back to the boundary. In that case fall
	// back to the leaf of the segment that preceded the boundary,
	// which lets the traversal continue through earlier compaction
	// segments within the same file.
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
				tail := r.CompactMetadata.PreservedSegment.TailUUID
				if _, visited := seen[tail]; visited {
					if prev, ok := boundaryLeaf[r.UUID]; ok {
						cur = prev
						continue
					}
					break
				}
				cur = tail
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

// readRecords reads all JSONL records from a single file.
func readRecords(path string) ([]rawRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
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
		return nil, err
	}
	return records, nil
}

// subagentMeta is the metadata from a subagent's .meta.json file.
type subagentMeta struct {
	AgentType   string `json:"agentType"`
	Description string `json:"description"`
	Name        string `json:"name"`
}

// parseSubagents reads all subagent JSONL files for a session and returns
// them as subagent values. The session directory is expected to be
// <projectDir>/<sessionID>/subagents/.
func parseSubagents(projectDir, sessionID string, extra bool) ([]subagent, error) {
	subDir := filepath.Join(projectDir, sessionID, "subagents")
	entries, err := os.ReadDir(subDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading subagents directory: %w", err)
	}

	var subs []subagent
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		agentID := strings.TrimSuffix(e.Name(), ".jsonl")
		agentID = strings.TrimPrefix(agentID, "agent-")

		var meta subagentMeta
		metaPath := filepath.Join(subDir, strings.TrimSuffix(e.Name(), ".jsonl")+".meta.json")
		if b, err := os.ReadFile(metaPath); err == nil {
			if err := json.Unmarshal(b, &meta); err != nil {
				log.Printf("warning: subagent %s meta: %v", agentID, err)
			}
		}

		path := filepath.Join(subDir, e.Name())
		records, err := readRecords(path)
		if err != nil {
			log.Printf("warning: subagent %s: %v", agentID, err)
			continue
		}

		// Build the chain. Subagent files are single-file sessions
		// without compaction, so a simple parentUuid walk suffices.
		byUUID := make(map[string]*rawRecord, len(records))
		for i := range records {
			r := &records[i]
			if r.UUID != "" {
				byUUID[r.UUID] = r
			}
		}

		// Find the leaf: the last record with a UUID and no child.
		children := make(map[string]bool, len(records))
		for _, r := range records {
			if r.ParentUUID != nil && *r.ParentUUID != "" {
				children[*r.ParentUUID] = true
			}
		}
		var leafUUID string
		for i := len(records) - 1; i >= 0; i-- {
			r := &records[i]
			if r.UUID != "" && !children[r.UUID] {
				leafUUID = r.UUID
				break
			}
		}
		if leafUUID == "" {
			continue
		}

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
			chain = append(chain, r)
			if r.ParentUUID == nil {
				break
			}
			cur = *r.ParentUUID
		}
		for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
			chain[i], chain[j] = chain[j], chain[i]
		}

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
		if len(msgs) == 0 {
			continue
		}

		subs = append(subs, subagent{
			ID:          agentID,
			Name:        meta.Name,
			Type:        meta.AgentType,
			Description: meta.Description,
			CreatedAt:   createdAt,
			Messages:    msgs,
		})
	}

	sort.Slice(subs, func(i, j int) bool {
		return subs[i].CreatedAt.Before(subs[j].CreatedAt)
	})

	return subs, nil
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
