import './styles.css'


const BoundingBox = () => {
const boxes = [
    { id: 1, top: '20%', left: '10%', width: '25%', height: '30%', color: 'box-blue', label: 'Person 0.95' },
    { id: 2, top: '50%', left: '60%', width: '20%', height: '25%', color: 'box-red', label: 'Safety Zone 0.90' },
];

return (
    <>
    {boxes.map(box => (
        <div
        key={box.id}
        className={`bounding-box ${box.color}`}
        style={{ top: box.top, left: box.left, width: box.width, height: box.height }}
        >
        <span className={`label ${box.color === 'box-blue' ? 'label-blue' : 'label-red'}`}>
            {box.label}
        </span>
        </div>
    ))}
    </>
  );
};

export default BoundingBox; 
