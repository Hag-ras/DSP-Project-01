import Button from "../Button/Button";
import "./Page.css";

export default function Page({ children }) {
  return (
    <div className="medical-container">
      <div className="monitor-section">{children}</div>

      <div className="btn-container">
        <Button bgColor="#4da6ff">Start</Button>
        <Button bgColor="#89cff0">Pause</Button>
        <Button bgColor="#ff4d6d">Stop</Button>
      </div>
    </div>
  );
}
