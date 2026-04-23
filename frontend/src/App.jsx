import { BrowserRouter } from "react-router-dom";
import Dashboard from "./dashboard/Dashboard";
import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { SnackbarProvider } from "notistack";

const App = () => {
  return (
    <BrowserRouter>
      <LocalizationProvider dateAdapter={AdapterDayjs}>
        <SnackbarProvider maxSnack={3}>
          <Dashboard />
        </SnackbarProvider>
      </LocalizationProvider>
    </BrowserRouter>
  );
};

export default App;
