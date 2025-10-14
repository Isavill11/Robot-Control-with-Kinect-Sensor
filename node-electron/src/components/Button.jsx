import "./Button.css"


function Button({ label, type = "default", onClick }){
  return (
    <button className={`custom-btn ${type}`} onClick={onClick}>
      {label}
    </button>
  );
}


export default Button; 
