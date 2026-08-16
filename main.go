// Command claude-export reads Claude CLI session history from its JSONL
// files and writes it as JSON or Markdown.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"
)

func main() {
	log.SetFlags(0)

	claudeDir := flag.String("claude-dir", defaultClaudeDir(), "path to Claude's config directory (contains projects/)")
	projectFilter := flag.String("project", "", "export only this project (substring match against path)")
	sessionFilter := flag.String("session", "", "export only sessions whose title contains this string (case-insensitive)")
	list := flag.Bool("list", false, "list all sessions and exit")
	extra := flag.Bool("extra", false, "include thinking blocks and tool calls")
	format := flag.String("format", "json", "output format: json or md")
	outFile := flag.String("o", "", "output file (default: stdout)")
	flag.Parse()

	projectsDir := filepath.Join(*claudeDir, "projects")
	dirs, err := discoverProjects(projectsDir)
	if err != nil {
		log.Fatalf("discovering projects: %v", err)
	}
	if len(dirs) == 0 {
		log.Fatal("no projects found")
	}

	if *list {
		var listErr error
		switch *format {
		case "json":
			listErr = listSessionsJSON(dirs, *projectFilter, *sessionFilter)
		default:
			listErr = listSessions(dirs, *projectFilter, *sessionFilter)
		}
		if listErr != nil {
			log.Fatal(listErr)
		}
		return
	}

	var exports []projectExport
	for _, dir := range dirs {
		ok, err := matchProject(dir, *projectFilter)
		if err != nil {
			log.Fatalf("failed to match project: %v", err)
		}
		if !ok {
			continue
		}
		export, err := exportProject(dir, *sessionFilter, *extra)
		if err != nil {
			log.Printf("warning: project %s: %v", filepath.Base(dir), err)
			continue
		}
		if len(export.Sessions) > 0 {
			exports = append(exports, export)
		}
	}

	if *format == "md" && len(exports) == 0 {
		return
	}

	var out io.Writer = os.Stdout
	if *outFile != "" {
		f, err := os.Create(*outFile)
		if err != nil {
			log.Fatalf("creating output file: %v", err)
		}
		defer func() {
			if err := f.Close(); err != nil {
				log.Fatalf("closing output file: %v", err)
			}
		}()
		out = f
	}

	switch *format {
	case "json":
		writeJSON(out, exports)
	case "md":
		writeMarkdown(out, exports)
	default:
		log.Fatalf("unknown format %q", *format)
	}
}

func defaultClaudeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude")
}

// projectExport holds all exported sessions for one project directory.
type projectExport struct {
	DirName  string    `json:"dirName"`
	Path     string    `json:"path,omitempty"`
	Sessions []session `json:"sessions"`
}

// session represents a single Claude CLI conversation.
type session struct {
	ID        string    `json:"id"`
	Title     string    `json:"title,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
	Messages  []message `json:"messages"`
}

// message is a single turn in a conversation.
type message struct {
	Role       string      `json:"role"`
	Timestamp  time.Time   `json:"timestamp"`
	Text       string      `json:"text,omitempty"`
	Thinking   string      `json:"thinking,omitempty"`
	Model      string      `json:"model,omitempty"`
	ToolUse    *toolUse    `json:"toolUse,omitempty"`
	ToolResult *toolResult `json:"toolResult,omitempty"`
}

// toolUse describes a tool invocation from an assistant turn.
type toolUse struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input,omitempty"`
}

// toolResult holds the output of a tool invocation.
type toolResult struct {
	ToolUseID string `json:"toolUseId"`
	Content   string `json:"content"`
}

func listSessions(dirs []string, projectFilter, sessionFilter string) error {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	for _, dir := range dirs {
		ok, err := matchProject(dir, projectFilter)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		files, err := sessionFiles(dir)
		if err != nil {
			log.Printf("warning: %v", err)
			continue
		}
		groups, err := groupSessionFiles(files)
		if err != nil {
			log.Printf("warning: grouping sessions: %v", err)
			continue
		}
		var lines []string
		var projectPath string
		for _, g := range groups {
			s, cwd, err := parseSession(g.files, g.id, false)
			if err != nil {
				continue
			}
			if projectPath == "" {
				projectPath = cwd
			}
			if !matchSession(s, sessionFilter) {
				continue
			}
			title := s.Title
			if title == "" {
				title = s.ID
			}
			created := ""
			if !s.CreatedAt.IsZero() {
				created = s.CreatedAt.Format(time.RFC3339)
			}
			lines = append(lines, fmt.Sprintf("  %s\t%s\t%s\n", title, s.ID, created))
		}
		if len(lines) == 0 {
			continue
		}
		name := projectPath
		if name == "" {
			name = filepath.Base(dir)
		}
		fmt.Fprintf(tw, "%s\t(%d sessions)\n", name, len(lines))
		for _, line := range lines {
			fmt.Fprint(tw, line)
		}
	}
	return tw.Flush()
}

func listSessionsJSON(dirs []string, projectFilter, sessionFilter string) error {
	type sessionInfo struct {
		Project   string    `json:"project"`
		DirName   string    `json:"dirName"`
		ID        string    `json:"id"`
		Title     string    `json:"title,omitempty"`
		CreatedAt time.Time `json:"createdAt"`
	}
	var infos []sessionInfo
	for _, dir := range dirs {
		ok, err := matchProject(dir, projectFilter)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		files, err := sessionFiles(dir)
		if err != nil {
			continue
		}
		groups, err := groupSessionFiles(files)
		if err != nil {
			continue
		}
		var projectPath string
		for _, g := range groups {
			s, cwd, err := parseSession(g.files, g.id, false)
			if err != nil {
				continue
			}
			if projectPath == "" {
				projectPath = cwd
			}
			if !matchSession(s, sessionFilter) {
				continue
			}
			infos = append(infos, sessionInfo{
				Project:   projectPath,
				DirName:   filepath.Base(dir),
				ID:        s.ID,
				Title:     s.Title,
				CreatedAt: s.CreatedAt,
			})
		}
	}
	b, err := json.Marshal(infos)
	if err != nil {
		return fmt.Errorf("encoding sessions list: %w", err)
	}
	fmt.Printf("%s\n", b)
	return nil
}

// exportProject loads and filters all sessions from a project directory.
func exportProject(dir, sessionFilter string, extra bool) (projectExport, error) {
	files, err := sessionFiles(dir)
	if err != nil {
		return projectExport{}, err
	}

	groups, err := groupSessionFiles(files)
	if err != nil {
		return projectExport{}, fmt.Errorf("grouping sessions: %w", err)
	}

	export := projectExport{DirName: filepath.Base(dir)}
	for _, g := range groups {
		s, cwd, err := parseSession(g.files, g.id, extra)
		if err != nil {
			log.Printf("warning: session %s: %v", g.id, err)
			continue
		}
		if export.Path == "" && cwd != "" {
			export.Path = cwd
		}
		if !matchSession(s, sessionFilter) {
			continue
		}
		if len(s.Messages) > 0 {
			export.Sessions = append(export.Sessions, s)
		}
	}

	sort.Slice(export.Sessions, func(i, j int) bool {
		return export.Sessions[i].CreatedAt.Before(export.Sessions[j].CreatedAt)
	})

	return export, nil
}

// matchProject reports whether dir matches pattern, checked against both
// the working directory from the project's sessions and the encoded
// directory name.
func matchProject(dir, pattern string) (bool, error) {
	if pattern == "" {
		return true, nil
	}
	p := strings.ToLower(pattern)
	if strings.Contains(strings.ToLower(filepath.Base(dir)), p) {
		return true, nil
	}
	cwd, err := projectCWD(dir)
	if err != nil {
		return false, err
	}
	return cwd != "" && strings.Contains(strings.ToLower(cwd), p), nil
}

// matchSession reports whether the session title or ID contains pattern.
func matchSession(s session, pattern string) bool {
	if pattern == "" {
		return true
	}
	p := strings.ToLower(pattern)
	return strings.Contains(strings.ToLower(s.Title), p) ||
		strings.Contains(strings.ToLower(s.ID), p)
}

func writeJSON(w io.Writer, exports []projectExport) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(exports); err != nil {
		log.Fatalf("writing JSON: %v", err)
	}
}

func writeMarkdown(w io.Writer, exports []projectExport) {
	var date time.Time
	for _, p := range exports {
		for _, s := range p.Sessions {
			if date.IsZero() || s.CreatedAt.Before(date) {
				date = s.CreatedAt
			}
		}
	}
	if !date.IsZero() {
		fmt.Fprintf(w, "---\ndate: %s\ntags:\n  - claude_chat\n---\n\n", date.Format(time.DateOnly))
	}

	for _, p := range exports {
		name := p.Path
		if name == "" {
			name = p.DirName
		}
		fmt.Fprintf(w, "# %s\n\n", name)
		for _, s := range p.Sessions {
			title := s.Title
			if title == "" {
				title = s.ID
			}
			fmt.Fprintf(w, "## %s\n\n", title)
			fmt.Fprintf(w, "- ID: %s\n", s.ID)
			fmt.Fprintf(w, "- Created: %s\n\n", s.CreatedAt.Format(time.RFC3339))
			for _, msg := range s.Messages {
				writeMarkdownMessage(w, msg)
			}
			fmt.Fprintln(w, "---")
		}
	}
}

func writeMarkdownMessage(w io.Writer, msg message) {
	ts := ""
	if !msg.Timestamp.IsZero() {
		ts = " — " + msg.Timestamp.Format(time.RFC3339)
	}

	if msg.ToolUse != nil {
		fmt.Fprintf(w, "### tool: %s%s\n\n", msg.ToolUse.Name, ts)
		if len(msg.ToolUse.Input) > 0 {
			var buf bytes.Buffer
			if json.Indent(&buf, msg.ToolUse.Input, "", "    ") == nil {
				fmt.Fprintf(w, "```json\n%s\n```\n\n", &buf)
			} else {
				fmt.Fprintf(w, "```json\n%s\n```\n\n", string(msg.ToolUse.Input))
			}
		}
		return
	}

	if msg.ToolResult != nil {
		fmt.Fprintf(w, "### tool_result: %s%s\n\n", msg.ToolResult.ToolUseID, ts)
		if msg.ToolResult.Content != "" {
			content := msg.ToolResult.Content
			if len(content) > 2000 {
				content = content[:2000] + "\n... (truncated)"
			}
			fmt.Fprintf(w, "```\n%s\n```\n\n", content)
		}
		return
	}

	heading := msg.Role
	if msg.Model != "" {
		heading += " (" + msg.Model + ")"
	}
	heading += ts
	fmt.Fprintf(w, "### %s\n\n", heading)

	if msg.Thinking != "" {
		fmt.Fprintf(w, "<details><summary>thinking</summary>\n\n%s\n\n</details>\n\n", msg.Thinking)
	}
	if msg.Text != "" {
		fmt.Fprintf(w, "%s\n\n", msg.Text)
	}
}
