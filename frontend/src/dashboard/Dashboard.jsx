import * as React from "react";

import { alpha } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import Stack from "@mui/material/Stack";
import { Navigate, Route, Routes } from "react-router-dom";

import AppNavbar from "./components/AppNavbar";
import SideMenu from "./components/SideMenu";
import DashboardContent from "./components/DashboardContent";
import ChoAnalysisRoute from "./components/ChoAnalysisRoute";
import ModalsHost from "./components/ModalsHost";
import AppTheme from "../shared-theme/AppTheme.jsx";
import BulkTestsPage from "./components/BulkTestsPage.jsx";
import DicomPullsPage from "./components/DicomPullsPage.jsx";
import DicomViewerPage from "./components/DicomViewerPage.jsx";
import DicomViewerRoute from "./components/DicomViewerRoute.jsx";
import {
  chartsCustomizations,
  dataGridCustomizations,
  datePickersCustomizations,
  treeViewCustomizations,
} from "./theme/customizations";
import StatusHeader from "./components/StatusHeader.jsx";
import { DashboardProvider } from "./context/DashboardContext";

const xThemeComponents = {
  ...chartsCustomizations,
  ...dataGridCustomizations,
  ...datePickersCustomizations,
  ...treeViewCustomizations,
};

export default function Dashboard(props) {
  return (
    <AppTheme {...props} themeComponents={xThemeComponents}>
      <CssBaseline enableColorScheme />
      <Box sx={{ display: "flex" }}>
        <SideMenu />
        <AppNavbar />
        <Box
          component='main'
          sx={(theme) => ({
            flexGrow: 1,
            backgroundColor: theme.vars
              ? `rgba(${theme.vars.palette.background.defaultChannel} / 1)`
              : alpha(theme.palette.background.default, 1),
            overflow: "auto",
          })}>
          <DashboardProvider>
            <Stack
              spacing={2}
              sx={{
                alignItems: "center",
                mx: 3,
                pb: 5,
                mt: { xs: 8, md: 0 },
              }}>
              {/* <StatusHeader /> */}
              <Container
                maxWidth='xl'
                sx={{
                  py: 4,
                  width: "100%",
                }}>
                <Routes>
                  {/* <Route path='/' element={<DashboardContent />} /> */}
                  {/* <Route path='/results' element={<DashboardContent />} /> */}
                  <Route
                    path='/'
                    element={<Navigate to='/main-dashboard' replace />}
                  />
                  <Route
                    path='/results/:seriesId'
                    element={<ChoAnalysisRoute />}
                  />
                  <Route path='/dicom-pulls' element={<DicomPullsPage />} />
                  <Route path='/main-dashboard' element={<BulkTestsPage />} />
                  <Route path='/dicom-viewer' element={<DicomViewerPage />} />
                  <Route
                    path='/dicom-viewer/:seriesId'
                    element={<DicomViewerRoute />}
                  />
                  <Route
                    path='*'
                    element={<Navigate to='/main-dashboard' replace />}
                  />
                </Routes>
              </Container>
              <ModalsHost />
            </Stack>
          </DashboardProvider>
        </Box>
      </Box>
    </AppTheme>
  );
}
