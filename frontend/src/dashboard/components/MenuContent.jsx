import * as React from "react";
import { Link as RouterLink, useLocation } from "react-router-dom";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import ContentPasteSearchIcon from "@mui/icons-material/ContentPasteSearch";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ScienceRoundedIcon from "@mui/icons-material/ScienceRounded";
import ViewInArRoundedIcon from "@mui/icons-material/ViewInArRounded";
import Tooltip from "@mui/material/Tooltip";

const mainNavItems = [
  {
    text: "Results",
    icon: ContentPasteSearchIcon,
    to: "/",
    isActive: (pathname) => pathname === "/" || pathname.startsWith("/results"),
  },
  {
    text: "Bulk Tests",
    icon: ScienceRoundedIcon,
    to: "/bulk-tests",
    isActive: (pathname) => pathname.startsWith("/bulk-tests"),
  },
  {
    text: "DICOM Pulls",
    icon: CloudDownloadRoundedIcon,
    to: "/dicom-pulls",
    isActive: (pathname) => pathname.startsWith("/dicom-pulls"),
  },
  {
    text: "DICOM Viewer",
    icon: ViewInArRoundedIcon,
    to: "/dicom-viewer",
    isActive: (pathname) => pathname.startsWith("/dicom-viewer"),
  },
];

export default function MenuContent() {
  const location = useLocation();

  return (
    <Box>
      <Stack spacing={0}>
        {mainNavItems.map((item) => {
          const Icon = item.icon;
          const selected = item.isActive
            ? item.isActive(location.pathname)
            : location.pathname === item.to;
          return (
            <Tooltip key={item.to} title={item.text} placement='right'>
              <ListItem key={item.to} sx={{ display: "block" }}>
                <ListItemButton
                  component={RouterLink}
                  to={item.to}
                  selected={selected}
                  sx={{
                    minHeight: 36,
                    minWidth: 36,
                    justifyContent: "center",
                  }}>
                  <ListItemIcon>
                    <Icon
                      sx={{
                        minHeight: 30,
                        minWidth: 30,
                      }}
                    />
                  </ListItemIcon>
                </ListItemButton>
              </ListItem>
            </Tooltip>
          );
        })}
      </Stack>
    </Box>
  );
}
