import { BrowserRouter } from "react-router-dom";
import Dashboard from "./dashboard/Dashboard";
import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";

const App = () => {
  return (
    <BrowserRouter>
      <LocalizationProvider dateAdapter={AdapterDayjs}>
        <Dashboard />
      </LocalizationProvider>
    </BrowserRouter>
  );
};

export default App;
