// DicomTable.tsx
import * as React from "react";
import {
  Box,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Tooltip,
  Typography,
  IconButton,
} from "@mui/material";
import KeyboardArrowRightRounded from "@mui/icons-material/KeyboardArrowRightRounded";
import KeyboardArrowDownRounded from "@mui/icons-material/KeyboardArrowDownRounded";

// ---------- Build hierarchical rows ----------
export function flattenDicomToTree(data) {
  const rows = [];
  let nextId = 0;

  const pushElem = (tag, elem, level, parentId) => {
    const VR = elem.VR || elem.Type || "";
    const isSeq = elem.Type === "Sequence" && Array.isArray(elem.Value);
    const hasChildren = isSeq && elem.Value.length > 0;

    const id = nextId++;
    let value;

    if (isSeq) {
      const childTags = new Set();
      for (const item of elem.Value)
        Object.keys(item || {}).forEach((t) => childTags.add(t));
      value = `[${[...childTags].map((t) => `(${t})`).join(", ")}]`;
    } else {
      const v = Array.isArray(elem.Value) ? elem.Value.join("\\") : elem.Value;
      value = v ?? "";
    }

    rows.push({
      id,
      parentId,
      level,
      hasChildren,
      Tag: tag,
      Attribute: elem.Name,
      Value: value,
      VR,
    });

    if (isSeq) {
      // Item rows and their children
      elem.Value.forEach((item, idx) => {
        const itemId = nextId++;
        rows.push({
          id: itemId,
          parentId: id,
          level: level + 1,
          hasChildren: Object.keys(item || {}).length > 0,
          isItem: true,
          Tag: "(fffe,e000)",
          Attribute: "Item",
          Value: `#${idx + 1}`,
          VR: "",
        });
        for (const [childTag, child] of Object.entries(item || {})) {
          pushElem(childTag, child, level + 2, itemId);
        }
      });
    }
  };

  for (const [tag, elem] of Object.entries(data)) {
    pushElem(tag, elem, 0, null);
  }

  return rows;
}

// ---------- Table component ----------
export default function DicomTable({ data }) {
  const allRows = React.useMemo(() => flattenDicomToTree(data), [data]);

  // Index rows by id and parent
  const byId = React.useMemo(() => {
    const m = new Map();
    allRows.forEach((r) => m.set(r.id, r));
    return m;
  }, [allRows]);

  // Expanded set for collapsible parents (only rows with children can expand)
  const [expanded, setExpanded] = React.useState(() => new Set());

  const toggle = (id) =>
    setExpanded((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  // A row is visible if all its ancestors are expanded (root has parentId=null)
  const isVisible = (row) => {
    let p = row.parentId;
    while (p !== null) {
      const parent = byId.get(p);
      if (parent.hasChildren && !expanded.has(parent.id)) return false;
      p = parent.parentId;
    }
    return true;
  };

  const visibleRows = allRows.filter(isVisible);

  return (
    <Paper sx={{ width: "100%", overflow: "auto" }}>
      <Box sx={{ maxWidth: 700, maxHeight: 520, overflow: "auto" }}>
        <Table size='small' stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ width: 140 }}>Tag</TableCell>
              <TableCell>Attribute</TableCell>
              <TableCell>Value</TableCell>
              <TableCell sx={{ width: 64, textAlign: "right" }}>VR</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {visibleRows.map((r) => (
              <TableRow key={r.id} hover>
                <TableCell
                  sx={{
                    fontFamily:
                      "ui-monospace, SFMono-Regular, Menlo, monospace",
                  }}>
                  {r.Tag}
                </TableCell>
                <TableCell>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      pl: r.level * 1.25,
                    }}>
                    {r.hasChildren ? (
                      <IconButton
                        size='small'
                        onClick={() => toggle(r.id)}
                        sx={{ mr: 0.5 }}
                        aria-label={expanded.has(r.id) ? "Collapse" : "Expand"}>
                        {expanded.has(r.id) ? (
                          <KeyboardArrowDownRounded fontSize='small' />
                        ) : (
                          <KeyboardArrowRightRounded fontSize='small' />
                        )}
                      </IconButton>
                    ) : (
                      <Box sx={{ width: 32, mr: 0.5 }} /> // spacer for alignment
                    )}
                    <Typography
                      component='span'
                      sx={{ fontWeight: r.isItem ? 500 : 400 }}>
                      {r.Attribute}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell sx={{ maxWidth: 420 }}>
                  <Tooltip title={String(r.Value)} disableInteractive>
                    <Typography noWrap>{String(r.Value)}</Typography>
                  </Tooltip>
                </TableCell>
                <TableCell
                  sx={{
                    textAlign: "right",
                    fontFamily: "ui-monospace, monospace",
                  }}>
                  {r.VR}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    </Paper>
  );
}
