import { useState } from "react";
import Button from "../Button/Button";
import "./Intro.css";

export default function Intro() {
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <div className="wrapper">
      <div className="modal">
        <button className="close-btn" onClick={() => setIsOpen(false)}>
          ×
        </button>
        <h1>Which Mode do you want?</h1>
        <div className="modal-buttons">
          <Button bgColor="#1e6091" to={'/medical'}>Medical</Button>
          <Button bgColor="#5ca8d8" to={'/sound'}>Sound</Button>
        </div>
      </div>
    </div>
  );
}
