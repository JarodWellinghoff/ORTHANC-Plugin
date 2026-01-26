import * as React from "react";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import buildMetadataSections from "./metadataSections.utils";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import Tooltip from "@mui/material/Tooltip";

const MetadataSections = ({ data, visibleSections }) => {
  const sections = React.useMemo(() => buildMetadataSections(data), [data]);

  const filteredSections = React.useMemo(() => {
    if (!visibleSections || visibleSections.length === 0) {
      return sections;
    }
    const allowed = new Set(visibleSections);
    return sections.filter((section) => allowed.has(section.title));
  }, [sections, visibleSections]);

  return (
    <Stack spacing={2} divider={<Divider flexItem />}>
      {filteredSections.map((section) => {
        if (section.items.length === 0) return null;
        return (
          <Box key={section.title}>
            <Typography
              variant='subtitle1'
              sx={{ fontWeight: 600, my: 1, textAlign: "center" }}>
              {section.title}
            </Typography>
            <Stack spacing={1.25}>
              {console.log(section.items)}
              {section.items.map((item) => (
                <Stack
                  key={item.key}
                  direction='row'
                  spacing={1}
                  justifyContent='space-between'
                  sx={{
                    alignItems: "baseline",
                  }}>
                  <Typography variant='body2' color='text.secondary'>
                    {item.label.pre
                      ? (item.label.pre.sup && (
                          <sup>{item.label.pre.sup}</sup>
                        )) ||
                        (item.label.pre.sub && <sub>{item.label.pre.sub}</sub>)
                      : null}
                    {item.label.main}
                    {item.label.post
                      ? (item.label.post.sup && (
                          <sup>{item.label.post.sup}</sup>
                        )) ||
                        (item.label.post.sub && (
                          <sub>{item.label.post.sub}</sub>
                        ))
                      : null}
                    {item.help_text && (
                      <Tooltip title={item.help_text} arrow>
                        <HelpOutlineIcon fontSize='small' sx={{ ml: 1 }} />
                      </Tooltip>
                    )}
                  </Typography>
                  <Stack
                    direction='row'
                    spacing={1}
                    sx={{
                      alignItems: "baseline",
                    }}>
                    <Typography variant='body2' sx={{ fontWeight: 500 }}>
                      {item.value}
                    </Typography>
                    {item.unit && (
                      <Typography
                        variant='body2'
                        color='text.secondary'
                        sx={{
                          marginLeft: 1,
                        }}>
                        {item.unit.pre
                          ? (item.unit.pre.sup && (
                              <sup>{item.unit.pre.sup}</sup>
                            )) ||
                            (item.unit.pre.sub && (
                              <sub>{item.unit.pre.sub}</sub>
                            ))
                          : null}
                        {item.unit.main}
                        {item.unit.post
                          ? (item.unit.post.sup && (
                              <sup>{item.unit.post.sup}</sup>
                            )) ||
                            (item.unit.post.sub && (
                              <sub>{item.unit.post.sub}</sub>
                            ))
                          : null}
                      </Typography>
                    )}
                  </Stack>
                </Stack>
              ))}
            </Stack>
          </Box>
        );
      })}
    </Stack>
  );
};

export default MetadataSections;
