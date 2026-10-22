/** Segmented skill gauge from tokens.css: `level` lit segments out of 12, the last lit one in gold at the max. */
export default function Gauge({ level, label }: { level: number; label: string }) {
  return (
    <div className="gauge" role="meter" aria-valuemin={0} aria-valuemax={12} aria-valuenow={level} aria-label={label}>
      {Array.from({ length: 12 }, (_, i) => (
        <i key={i} className={i < level ? `on${i === level - 1 && level >= 11 ? ' top' : ''}` : ''} />
      ))}
    </div>
  );
}
