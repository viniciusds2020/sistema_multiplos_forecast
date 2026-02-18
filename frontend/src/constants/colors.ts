export const COLORS = {
  primary: '#4361ee',
  secondary: '#3a0ca3',
  success: '#06d6a0',
  danger: '#ef476f',
  warning: '#ffd166',
  info: '#118ab2',
  dark: '#1a1a2e',
  dark_secondary: '#16213e',
  light: '#f8f9fa',
  white: '#ffffff',
  text: '#2b2d42',
  text_muted: '#8d99ae',
} as const;

export const MODEL_COLORS: Record<string, string> = {
  XGBoost: '#4361ee',
  LightGBM: '#06d6a0',
  AutoARIMA: '#ef476f',
  Prophet: '#ffd166',
  Chronos: '#118ab2',
  CrostonSBA: '#7209b7',
  Agregado: '#3a0ca3',
};
