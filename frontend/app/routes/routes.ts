import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./home";

const AppRoutes: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </Router>
  );
};

export default AppRoutes;
