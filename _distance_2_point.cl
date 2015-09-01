inline float _distance_2_point(float x, float y, float rx, float ry, float sw, float sh, float ez, float ex, float ey)
{
  float dx = x / rx * sw - sw / 2.0f + ex;
  float dy = y / ry * sh - sh / 2.0f - ey;
  return sqrt(ez*ez + dx*dx + dy*dy);
}
