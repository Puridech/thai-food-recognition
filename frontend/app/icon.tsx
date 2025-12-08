import { ImageResponse } from 'next/og';

export const runtime = 'edge';

export const size = {
  width: 32,
  height: 32,
};

export const contentType = 'image/png';

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #FF6B35 0%, #F7931E 100%)',
          borderRadius: '50%',
        }}
      >
        {/* Simple Bowl Icon */}
        <div
          style={{
            width: '20px',
            height: '20px',
            borderRadius: '50% 50% 50% 50% / 40% 40% 60% 60%',
            background: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {/* Steam lines */}
          <div
            style={{
              display: 'flex',
              gap: '2px',
              transform: 'translateY(-8px)',
            }}
          >
            <div style={{ width: '2px', height: '6px', background: 'white', borderRadius: '1px' }} />
            <div style={{ width: '2px', height: '8px', background: 'white', borderRadius: '1px' }} />
            <div style={{ width: '2px', height: '6px', background: 'white', borderRadius: '1px' }} />
          </div>
        </div>
      </div>
    ),
    {
      ...size,
    }
  );
}
