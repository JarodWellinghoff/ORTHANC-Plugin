import { Link as RouterLink, useLocation } from "react-router-dom";
import { Box, Stack, ListItem, ListItemButton, ListItemIcon, Tooltip } from "@mui/material";
import HomeRoundedIcon from "@mui/icons-material/HomeRounded";
import ContentPasteSearchIcon from "@mui/icons-material/ContentPasteSearch";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ScienceRoundedIcon from "@mui/icons-material/ScienceRounded";
import InsightsRoundedIcon from "@mui/icons-material/InsightsRounded";
const mainNavItems = [
  { text: "Home", icon: HomeRoundedIcon, to: "/home" },
  { text: "DICOM Query and Pulls", icon: CloudDownloadRoundedIcon, to: "/dicom-pulls" },
  { text: "Batch Analysis", icon: ScienceRoundedIcon, to: "/main-dashboard" },
  { text: "Result View/Modification of Individual Exam", icon: ContentPasteSearchIcon, to: "/results" },
  { text: "Statistics Report", icon: InsightsRoundedIcon, to: "/plotting" },
];
export default function MenuContent() {
  const { pathname } = useLocation();
  return <Box><Stack spacing={0}>{mainNavItems.map(({ text, to, icon: Icon }) => <Tooltip key={to} title={text} placement='right'>
    <ListItem sx={{ display: "block" }}><ListItemButton component={RouterLink} to={to} aria-label={text}
      selected={pathname.startsWith(to) || (to === "/home" && pathname === "/")}
      sx={{ minHeight: 36, minWidth: 36, justifyContent: "center" }}>
      <ListItemIcon><Icon sx={{ minHeight: 30, minWidth: 30 }} /></ListItemIcon>
    </ListItemButton></ListItem>
  </Tooltip>)}</Stack></Box>;
}
