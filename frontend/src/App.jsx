import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Dashboard from "./dashboard/Dashboard";
import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { SnackbarProvider } from "notistack";

// useBlocker requires a data router. Keep the existing Dashboard routes and
// providers, and create the router once (not during a component render).
const router = createBrowserRouter([
  {
    path: "*",
    element: (
      <LocalizationProvider dateAdapter={AdapterDayjs}>
        <SnackbarProvider maxSnack={3}>
          <Dashboard />
        </SnackbarProvider>
      </LocalizationProvider>
    ),
  },
]);

const App = () => <RouterProvider router={router} />;

export default App;
