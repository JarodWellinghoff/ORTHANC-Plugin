import { styled } from "@mui/material/styles";
import MuiDrawer, { drawerClasses } from "@mui/material/Drawer";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import MenuContent from "./MenuContent";
import OptionsMenu from "./OptionsMenu";

export const drawerWidth = 65;
export const userSectionHeight = 64;

const Drawer = styled(MuiDrawer)(() => ({
  width: drawerWidth,
  flexShrink: 0,
  whiteSpace: "nowrap",
  boxSizing: "border-box",
  overflowX: "hidden",
  ["& .MuiDrawer-paper"]: {
    width: drawerWidth,
  },
}));

export default function SideMenu() {
  return (
    <Drawer
      variant='permanent'
      sx={{
        display: { xs: "none", md: "block" },
        [`& .${drawerClasses.paper}`]: {
          backgroundColor: "background.paper",
        },
      }}>
      <Box
        sx={{
          display: "flex",
          mt: "calc(var(--template-frame-height, 0px) + 4px)",
          p: 1.5,
        }}></Box>
      <Divider />
      <Box
        sx={{
          overflowX: "hidden",
          height: "100%",
          display: "flex",
          flexDirection: "column",
        }}>
        <MenuContent />
      </Box>
      <Box
        sx={{
          p: 2,
          gap: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderTop: "1px solid",
          borderColor: "divider",
          height: userSectionHeight,
        }}>
        <OptionsMenu />
      </Box>
    </Drawer>
  );
}
