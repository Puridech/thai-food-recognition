interface LogoProps {
  className?: string;
  size?: number;
}

export default function Logo({ className = '', size = 40 }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FF6B35" />
          <stop offset="100%" stopColor="#F7931E" />
        </linearGradient>
      </defs>

      {/* Background Circle */}
      <circle cx="50" cy="50" r="48" fill="url(#grad1)" />
      
      {/* Bowl - Simple white circle */}
      <circle cx="50" cy="55" r="28" fill="#FFFFFF" />
      
      {/* Bowl shadow inside */}
      <ellipse cx="50" cy="55" rx="24" ry="20" fill="#F5F5F5" />
      
      {/* Food inside - Orange/Yellow */}
      <circle cx="50" cy="54" r="18" fill="#FFB800" />
      
      {/* Highlights on food */}
      <circle cx="45" cy="50" r="4" fill="#FFD700" opacity="0.8" />
      <circle cx="55" cy="52" r="3" fill="#FFD700" opacity="0.8" />
      
      {/* Steam - Simple curved lines */}
      <path
        d="M 40 38 Q 38 30, 40 22"
        stroke="#FFFFFF"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        opacity="0.9"
      />
      
      <path
        d="M 50 35 Q 50 27, 50 18"
        stroke="#FFFFFF"
        strokeWidth="3.5"
        strokeLinecap="round"
        fill="none"
        opacity="0.9"
      />
      
      <path
        d="M 60 38 Q 62 30, 60 22"
        stroke="#FFFFFF"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        opacity="0.9"
      />
    </svg>
  );
}
