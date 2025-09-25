import { Route, Routes, useLocation } from "react-router";
import Intro from "./components/Intro/Intro";
import Navbar from "./components/Navbar/Navbar";
import Page from "./components/Page/Page";

export default function App() {
  const location = useLocation();

  const showNavbar = location.pathname !== "/";
  return (
    <>
      {showNavbar && <Navbar />}
      <Routes>
        <Route path="/" element={<Intro />} />
        <Route path="/medical" element={<Page>ECG|EEG</Page>} />
        <Route path="/sound" element={<Page>Sound Graph</Page>} />
      </Routes>
    </>
  );
}
