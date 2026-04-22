import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          background: "linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)",
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {/* Scales of justice SVG */}
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#60a5fa"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          {/* Center pole */}
          <line x1="12" y1="3" x2="12" y2="21" />
          {/* Base */}
          <line x1="8" y1="21" x2="16" y2="21" />
          {/* Crossbar */}
          <line x1="3" y1="7" x2="21" y2="7" />
          {/* Left pan strings */}
          <line x1="3" y1="7" x2="6" y2="14" />
          <line x1="9" y1="7" x2="6" y2="14" />
          {/* Right pan strings */}
          <line x1="15" y1="7" x2="18" y2="14" />
          <line x1="21" y1="7" x2="18" y2="14" />
          {/* Left pan arc */}
          <path d="M3 14 Q6 17 9 14" />
          {/* Right pan arc */}
          <path d="M15 14 Q18 17 21 14" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
