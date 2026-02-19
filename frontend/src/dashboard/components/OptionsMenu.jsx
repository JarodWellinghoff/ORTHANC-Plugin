import * as React from "react";
import { styled } from "@mui/material/styles";
import Divider from "@mui/material/Divider";
import Menu from "@mui/material/Menu";
import MuiMenuItem from "@mui/material/MenuItem";
import ListItemText from "@mui/material/ListItemText";
import ListItemIcon from "@mui/material/ListItemIcon";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";
import MenuButton from "./MenuButton";
import Avatar from "@mui/material/Avatar";
import HelpRoundedIcon from "@mui/icons-material/HelpRounded";
import SettingsRoundedIcon from "@mui/icons-material/SettingsRounded";
import InfoRoundedIcon from "@mui/icons-material/InfoRounded";
import ColorModeIconDropdown from "../../shared-theme/ColorModeIconDropdown";

const MenuItem = styled(MuiMenuItem)({
  margin: "2px 0",
});

const topListItems = [
  { text: "Profile", icon: InfoRoundedIcon },
  { text: "Settings", icon: SettingsRoundedIcon },
  { text: "Feedback", icon: HelpRoundedIcon },
  { text: "", icon: ColorModeIconDropdown },
];

const bottomListItems = [{ text: "Logout", icon: LogoutRoundedIcon }];

export default function OptionsMenu() {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };
  return (
    <>
      <MenuButton
        aria-label='Open menu'
        onClick={handleClick}
        sx={{ border: "none" }}>
        <Avatar
          sizes='small'
          alt='John Doe'
          src='/static/images/avatar/7.jpg'
          sx={{ width: 36, height: 36 }}
        />
      </MenuButton>
      <Menu
        anchorEl={anchorEl}
        id='menu'
        open={open}
        onClose={handleClose}
        onClick={handleClose}
        anchorOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "bottom",
          horizontal: "left",
        }}>
        {topListItems.map((item) => {
          const Icon = item.icon;
          return (
            <MenuItem onClick={handleClose}>
              <ListItemIcon>
                <Icon fontSize='small' />
              </ListItemIcon>
              <ListItemText>{item.text}</ListItemText>
            </MenuItem>
          );
        })}
        <Divider />
        {bottomListItems.map((item) => {
          const Icon = item.icon;
          return (
            <MenuItem onClick={handleClose}>
              <ListItemIcon>
                <Icon fontSize='small' />
              </ListItemIcon>
              <ListItemText>{item.text}</ListItemText>
            </MenuItem>
          );
        })}
      </Menu>
    </>
  );
}
