import * as React from "react";
import { useNavigate } from "react-router-dom";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActionArea from "@mui/material/CardActionArea";
import Chip from "@mui/material/Chip";
import Button from "@mui/material/Button";
import { alpha } from "@mui/material/styles";

import ScienceRoundedIcon from "@mui/icons-material/ScienceRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ViewInArRoundedIcon from "@mui/icons-material/ViewInArRounded";
import BiotechRoundedIcon from "@mui/icons-material/BiotechRounded";
import AssessmentRoundedIcon from "@mui/icons-material/AssessmentRounded";
import LocalHospitalRoundedIcon from "@mui/icons-material/LocalHospitalRounded";
import SpeedRoundedIcon from "@mui/icons-material/SpeedRounded";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";

const navCards = [
  {
    title: "Bulk Tests",
    description:
      "Browse the analyzed CT series catalog, filter by patient, protocol, or institute, and queue CHO analyses across many series at once.",
    icon: ScienceRoundedIcon,
    to: "/main-dashboard",
    accent: "primary",
  },
  {
    title: "DICOM Pulls",
    description:
      "Query remote PACS modalities via C-FIND and retrieve studies with C-MOVE. Schedule pull batches and monitor their progress.",
    icon: CloudDownloadRoundedIcon,
    to: "/dicom-pulls",
    accent: "success",
  },
  {
    title: "DICOM Viewer",
    description:
      "Inspect series in a built-in multi-planar reconstruction (MPR) viewer powered by Cornerstone for axial, coronal, and sagittal views.",
    icon: ViewInArRoundedIcon,
    to: "/dicom-viewer",
    accent: "info",
  },
];

const capabilities = [
  {
    title: "CHO Detectability",
    description:
      "Channelized Hotelling Observer model computes d-prime for low-contrast lesion detectability across reconstructions.",
    icon: BiotechRoundedIcon,
  },
  {
    title: "Noise & Resolution",
    description:
      "Noise Power Spectrum (NPS) and presampling MTF — including MTF50 and MTF10 — measured directly from each CT series.",
    icon: AssessmentRoundedIcon,
  },
  {
    title: "Dose Metrics",
    description:
      "CTDIvol, body-part-aware SSDE, and DLP recorded per series alongside detectability for full quality and dose context.",
    icon: LocalHospitalRoundedIcon,
  },
  {
    title: "End-to-End Pipeline",
    description:
      "Pulls from Orthanc, runs analysis with caching, persists to PostgreSQL, and serves results to the dashboard in real time.",
    icon: SpeedRoundedIcon,
  },
];

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <Stack spacing={5} sx={{ pb: 4 }}>
      {/* Hero */}
      <Box
        sx={(theme) => ({
          position: "relative",
          overflow: "hidden",
          px: { xs: 3, md: 6 },
          py: { xs: 4, md: 7 },
          borderRadius: 3,
          border: "1px solid",
          borderColor: "divider",
          backgroundImage: `linear-gradient(135deg, ${alpha(
            theme.palette.primary.main,
            0.14,
          )} 0%, ${alpha(theme.palette.primary.dark, 0.04)} 100%)`,
        })}>
        <Stack spacing={2.5} sx={{ maxWidth: 880, position: "relative" }}>
          <Chip
            label='ORTHANC · CHO Analysis Platform'
            color='primary'
            variant='outlined'
            size='small'
            sx={{ alignSelf: "flex-start", fontWeight: 500 }}
          />
          <Typography
            variant='h3'
            component='h1'
            sx={{
              fontWeight: 600,
              letterSpacing: "-0.02em",
              fontSize: { xs: "2rem", md: "2.75rem" },
            }}>
            Welcome to the CHO Analysis Platform
          </Typography>
          <Typography
            variant='body1'
            color='text.secondary'
            sx={{ fontSize: "1.05rem", lineHeight: 1.65 }}>
            A research and clinical workbench for CT image quality analysis. The
            platform combines an Orthanc DICOM server, PostgreSQL-backed result
            storage, and a Channelized Hotelling Observer pipeline to quantify
            lesion detectability, noise, resolution, and dose across your CT
            series.
          </Typography>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={1.5}
            sx={{ pt: 1 }}>
            <Button
              variant='contained'
              size='large'
              endIcon={<ArrowForwardRoundedIcon />}
              onClick={() => navigate("/main-dashboard")}>
              Go to Bulk Tests
            </Button>
            <Button
              variant='outlined'
              size='large'
              onClick={() => navigate("/dicom-pulls")}>
              Pull DICOM Studies
            </Button>
          </Stack>
        </Stack>
      </Box>

      {/* Quick navigation cards */}
      <Box>
        <Typography variant='h5' sx={{ fontWeight: 600, mb: 0.5 }}>
          Get started
        </Typography>
        <Typography variant='body2' color='text.secondary' sx={{ mb: 3 }}>
          Jump to the area you want to work in.
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" },
            gap: 2.5,
          }}>
          {navCards.map((card) => {
            const Icon = card.icon;
            return (
              <Card
                key={card.to}
                variant='outlined'
                sx={{
                  height: "100%",
                  transition:
                    "transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease",
                  "&:hover": {
                    transform: "translateY(-3px)",
                    boxShadow: 4,
                    borderColor: (theme) => theme.palette[card.accent].main,
                  },
                }}>
                <CardActionArea
                  onClick={() => navigate(card.to)}
                  sx={{ height: "100%", p: 3 }}>
                  <Stack spacing={2} sx={{ height: "100%" }}>
                    <Box
                      sx={(theme) => ({
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        backgroundColor: alpha(
                          theme.palette[card.accent].main,
                          0.12,
                        ),
                        color: theme.palette[card.accent].main,
                      })}>
                      <Icon sx={{ fontSize: 28 }} />
                    </Box>
                    <Typography variant='h6' sx={{ fontWeight: 600 }}>
                      {card.title}
                    </Typography>
                    <Typography
                      variant='body2'
                      color='text.secondary'
                      sx={{ flexGrow: 1, lineHeight: 1.6 }}>
                      {card.description}
                    </Typography>
                    <Stack
                      direction='row'
                      spacing={0.5}
                      alignItems='center'
                      sx={{ color: `${card.accent}.main`, mt: "auto" }}>
                      <Typography variant='button' sx={{ fontWeight: 600 }}>
                        Open
                      </Typography>
                      <ArrowForwardRoundedIcon fontSize='small' />
                    </Stack>
                  </Stack>
                </CardActionArea>
              </Card>
            );
          })}
        </Box>
      </Box>

      {/* Capabilities */}
      <Box>
        <Typography variant='h5' sx={{ fontWeight: 600, mb: 0.5 }}>
          What this platform measures
        </Typography>
        <Typography variant='body2' color='text.secondary' sx={{ mb: 3 }}>
          Each analyzed series produces a consistent set of image-quality and
          dose metrics.
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr",
              sm: "repeat(2, 1fr)",
              lg: "repeat(4, 1fr)",
            },
            gap: 2,
          }}>
          {capabilities.map((cap) => {
            const Icon = cap.icon;
            return (
              <Card key={cap.title} variant='outlined' sx={{ height: "100%" }}>
                <CardContent>
                  <Stack spacing={1.5}>
                    <Icon color='primary' sx={{ fontSize: 26 }} />
                    <Typography variant='subtitle1' sx={{ fontWeight: 600 }}>
                      {cap.title}
                    </Typography>
                    <Typography
                      variant='body2'
                      color='text.secondary'
                      sx={{ lineHeight: 1.55 }}>
                      {cap.description}
                    </Typography>
                  </Stack>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      </Box>

      {/* Footer note */}
      <Box
        sx={(theme) => ({
          p: 2.5,
          borderRadius: 2,
          border: "1px dashed",
          borderColor: "divider",
          backgroundColor: alpha(theme.palette.background.default, 0.4),
        })}>
        <Typography variant='body2' color='text.secondary'>
          New here? Start by pulling a few studies from your PACS in{" "}
          <Box component='span' sx={{ fontWeight: 600, color: "text.primary" }}>
            DICOM Pulls
          </Box>
          , then head over to{" "}
          <Box component='span' sx={{ fontWeight: 600, color: "text.primary" }}>
            Bulk Tests
          </Box>{" "}
          to filter the series catalog and queue analyses.
        </Typography>
      </Box>
    </Stack>
  );
};

export default HomePage;
