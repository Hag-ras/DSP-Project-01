import "./navbar.css";
export default function Navbar() {
  return (
    <nav>
      <Select data={["ECG", "EEG"]} />

      <div>
        <Select name="lead" />

        <Select />
      </div>
    </nav>
  );
}

function Select({ data = [], name = "option", size = 3, defaultValue = "" }) {
  const options =
    data.length > 0
      ? data
      : Array.from({ length: size }, (_, i) => `${name} ${i + 1}`);

  return (
    <select defaultValue={defaultValue}>
      <option value="" disabled>
        Select {name}
      </option>
      {options.map((item, i) => (
        <option key={i} value={item}>
          {item}
        </option>
      ))}
    </select>
  );
}
