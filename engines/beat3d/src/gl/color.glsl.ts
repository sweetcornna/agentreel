// Colour science shared by every shader: AgX display transform, sRGB transfer, grading.
// Working space everywhere is linear Rec.709/sRGB primaries, scene-referred (HDR, unbounded).

export const COLOR_GLSL = /* glsl */ `
#ifndef B3D_COLOR
#define B3D_COLOR
float luma709(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
vec3 linearToSrgb(vec3 c) {
  c = max(c, vec3(0.0));
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

// AgX (Troy Sobotka), sigmoid fitted by a 6th-order polynomial. Input: linear scene-referred.
vec3 agxContrast(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
/** Scene linear -> AgX log-encoded sigmoid output (display-encoded, ~gamma 2.2). */
vec3 agxEncode(vec3 v) {
  const mat3 inset = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  const float minEv = -12.47393;
  const float maxEv = 4.026069;
  v = inset * max(v, vec3(1e-10));
  v = clamp(log2(v), minEv, maxEv);
  v = (v - minEv) / (maxEv - minEv);
  return agxContrast(v);
}
/** ASC-CDL style look applied in AgX's encoded space (how AgX "looks" are defined). */
vec3 agxLook(vec3 v, vec3 slope, vec3 offset, vec3 power, float sat) {
  float l = luma709(v);
  v = pow(max(v * slope + offset, vec3(0.0)), power);
  return l + sat * (v - l);
}
/** AgX encoded -> display linear. */
vec3 agxDecode(vec3 v) {
  const mat3 outset = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  v = outset * v;
  return pow(max(v, vec3(0.0)), vec3(2.2));
}

/** Classic lift/gamma/gain on display-linear values in [0,1]. */
vec3 liftGammaGain(vec3 c, vec3 lift, vec3 gamma, vec3 gain) {
  c = clamp(c, 0.0, 1.0);
  c = gain * (c + lift * (1.0 - c));
  return pow(max(c, vec3(0.0)), 1.0 / max(gamma, vec3(1e-3)));
}
vec3 saturate709(vec3 c, float s) {
  float l = luma709(c);
  return max(vec3(l) + s * (c - l), vec3(0.0));
}
#endif
`;
