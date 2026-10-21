// WebGL2 renderer: all nodes in one instanced draw (camera-facing billboards
// sampling a mipmapped texture-array atlas), all edges in one GL_LINES draw.

const NODE_VS = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec4 aPos;     // xyz, size
layout(location=2) in vec2 aMeta;    // tile (<0 = hub), state
layout(location=3) in vec3 aMean;    // mosaic: the post's mean colour
layout(location=4) in vec3 aRect;    // mosaic: cell in the target image (u0, v0, size)
uniform mat4 uViewProj;
uniform vec3 uRight, uUp, uEye;
uniform vec2 uFog;
uniform float uPerRow, uPerPage, uInset;
out vec2 vCorner;
out vec3 vTex;
out float vState, vHub, vFog;
out vec3 vMean, vRect;
void main() {
  vMean = aMean;
  vRect = aRect;
  vec3 wp = aPos.xyz + (uRight * aCorner.x + uUp * aCorner.y) * aPos.w;
  gl_Position = uViewProj * vec4(wp, 1.0);
  vCorner = aCorner;
  float t = aMeta.x;
  vHub = t < 0.0 ? 1.0 : 0.0;
  float page = floor(t / uPerPage);
  float local = t - page * uPerPage;
  vec2 cell = vec2(mod(local, uPerRow), floor(local / uPerRow));
  vec2 uv = vec2(aCorner.x, -aCorner.y) * (1.0 - 2.0 * uInset) + 0.5;  // inset avoids mip bleeding
  vTex = vec3((cell + uv) / uPerRow, page);
  vState = aMeta.y;
  vFog = smoothstep(uFog.x, uFog.y, distance(aPos.xyz, uEye));
}`;

const NODE_FS = `#version 300 es
precision highp float;
precision highp sampler2DArray;
uniform sampler2DArray uTex;
uniform float uHasTex;
uniform vec3 uBg, uHubColor;
uniform sampler2D uTarget;
uniform float uTint, uRound;
in vec2 vCorner;
in vec3 vMean, vRect;
in vec3 vTex;
in float vState, vHub, vFog;
out vec4 o;
float sdRoundBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
void main() {
  float d = vHub > 0.5 ? length(vCorner) - 0.5 : sdRoundBox(vCorner, vec2(0.5), uRound);
  float w = fwidth(d);
  float cov = clamp(0.5 - d / w, 0.0, 1.0);
  if (cov <= 0.0) discard;
  vec3 col;
  if (vHub > 0.5) {
    float r = length(vCorner) * 2.0;
    col = uHubColor * (1.2 - 0.6 * r * r);
    if (vState > 0.5) col = mix(col, vec3(1.0), 0.5);
  } else {
    col = uHasTex > 0.5 ? texture(uTex, vTex).rgb : vec3(0.22);
    // photomosaic colour mapping, per pixel tone transfer: the target pixel plus the
    // tile's own detail (its deviation from its mean), with the detail fading as tint rises
    if (uTint > 0.0) {
      vec3 t = texture(uTarget, vRect.xy + (vec2(vCorner.x, -vCorner.y) + 0.5) * vRect.z).rgb;
      vec3 mapped = t + (col - vMean) * (1.0 - 0.7 * uTint);
      col = clamp(mix(col, mapped, uTint), 0.0, 1.0);
    }
    // highlight ring
    if (vState > 0.5) col = mix(col, vec3(1.0), smoothstep(-0.06 - w, -0.06, d));
  }
  if (vState < -0.5) col *= 0.12;
  col = mix(col, uBg, vFog * 0.8);
  o = vec4(col, cov);
}`;

const EDGE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in float aState;
uniform mat4 uViewProj;
uniform vec3 uEye;
uniform vec2 uFog;
uniform float uAlpha;
out float vA;
void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  float fog = 1.0 - 0.85 * smoothstep(uFog.x, uFog.y, distance(aPos, uEye));
  vA = (aState > 0.5 ? 0.55 : aState < -0.5 ? 0.15 * uAlpha : uAlpha) * fog;
}`;

const EDGE_FS = `#version 300 es
precision mediump float;
uniform vec3 uEdgeColor;
in float vA;
out vec4 o;
void main() { o = vec4(uEdgeColor * vA, 1.0); }`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
    const name = gl.getActiveUniform(p, i).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}

export class Renderer {
  constructor(canvas, cap) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl; this.canvas = canvas; this.cap = cap;
    this.node = program(gl, NODE_VS, NODE_FS);
    this.edge = program(gl, EDGE_VS, EDGE_FS);
    this.bg = [0.043, 0.043, 0.055];
    this.hasTex = 0;

    // node VAO: static corner quad + per-instance pos/size (dynamic) + tile/state
    this.pos = new Float32Array(cap * 4);
    this.meta = new Float32Array(cap * 2);
    this.mean = new Float32Array(cap * 3);
    this.rect = new Float32Array(cap * 3);
    this.nodeVao = gl.createVertexArray();
    gl.bindVertexArray(this.nodeVao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-.5, -.5, .5, -.5, -.5, .5, .5, .5]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.pos.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    this.metaBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.metaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.meta.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(2, 1);
    this.meanBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meanBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.mean.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(3, 1);
    this.rectBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.rectBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.rect.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(4, 1);

    this.edgeVao = gl.createVertexArray();
    this.edgePosBuf = gl.createBuffer();
    this.edgeStateBuf = gl.createBuffer();
    gl.bindVertexArray(null);
  }

  async loadAtlas(atlas) {
    const gl = this.gl;
    const imgs = await Promise.all(atlas.pages.map(src => new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = src;
    })));
    const size = imgs[0].width;
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.tex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, Math.log2(size) + 1, gl.RGBA8, size, size, imgs.length);
    imgs.forEach((im, i) => gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, size, size, 1, gl.RGBA, gl.UNSIGNED_BYTE, im));
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    if (aniso) gl.texParameterf(gl.TEXTURE_2D_ARRAY, aniso.TEXTURE_MAX_ANISOTROPY_EXT, 4);
    this.perRow = atlas.perRow;
    this.inset = 2 / atlas.tile;
    this.hasTex = 1;
  }

  // edges: Int32Array src/dst into node arrays
  setEdges(src, dst) {
    const gl = this.gl;
    this.src = src; this.dst = dst;
    this.edgeCount = src.length;
    this.edgePos = new Float32Array(this.edgeCount * 6);
    this.edgeState = new Float32Array(this.edgeCount * 2);
    gl.bindVertexArray(this.edgeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgePosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgePos.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeStateBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeState.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  // per-node tile index (<0 hub) and state (-1 dim, 0, 1 highlight); per-edge state
  setStates(n, tile, nodeState, edgeState) {
    const gl = this.gl;
    for (let i = 0; i < n; i++) { this.meta[i * 2] = tile[i]; this.meta[i * 2 + 1] = nodeState[i]; }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.metaBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.meta, 0, n * 2);
    for (let e = 0; e < this.edgeCount; e++) this.edgeState[e * 2] = this.edgeState[e * 2 + 1] = edgeState[e];
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeStateBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.edgeState);
  }

  // mosaic per-tile data: post mean colour (0..1) and cell rect in the target image
  setMosaic(mean, rect) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meanBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, mean);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.rectBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, rect);
  }

  setTarget(img) {
    const gl = this.gl;
    this.target = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.target);
    // centre-crop to square, like the mosaic analysis
    const s = Math.min(img.width, img.height), cv = document.createElement('canvas');
    cv.width = cv.height = s;
    cv.getContext('2d').drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, s, s);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  resize() {
    const dpr = Math.min(devicePixelRatio, 2, this.dprCap ?? 2);
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  draw(cam, layout, sizes, opts) {
    const gl = this.gl, n = layout.n;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(...this.bg, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const { x, y, z } = layout;
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      this.pos[o] = x[i]; this.pos[o + 1] = y[i]; this.pos[o + 2] = z[i]; this.pos[o + 3] = sizes[i];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.pos, 0, n * 4);

    // nodes: opaque with alpha-to-coverage, so no depth sorting is needed
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    const N = this.node;
    gl.useProgram(N.p);
    gl.uniformMatrix4fv(N.u.uViewProj, false, cam.viewProj);
    gl.uniform3fv(N.u.uRight, cam.right);
    gl.uniform3fv(N.u.uUp, cam.up);
    gl.uniform3fv(N.u.uEye, cam.eye);
    gl.uniform2f(N.u.uFog, cam.fogNear, cam.fogFar);
    gl.uniform1f(N.u.uPerRow, this.perRow ?? 16);
    gl.uniform1f(N.u.uPerPage, (this.perRow ?? 16) ** 2);
    gl.uniform1f(N.u.uInset, this.inset ?? 0);
    gl.uniform1f(N.u.uHasTex, this.hasTex);
    gl.uniform3fv(N.u.uBg, this.bg);
    gl.uniform3fv(N.u.uHubColor, opts.hubColor);
    gl.uniform1f(N.u.uTint, opts.tint);
    gl.uniform1f(N.u.uRound, opts.round);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.tex ?? null);
    gl.uniform1i(N.u.uTex, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.target ?? null);
    gl.uniform1i(N.u.uTarget, 1);
    gl.bindVertexArray(this.nodeVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
    gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);

    // edges after nodes: depth-tested, additive, no depth writes
    if (opts.edges && this.edgeCount) {
      const ep = this.edgePos, s = this.src, d = this.dst;
      for (let e = 0; e < this.edgeCount; e++) {
        const a = s[e], b = d[e], o = e * 6;
        ep[o] = x[a]; ep[o + 1] = y[a]; ep[o + 2] = z[a];
        ep[o + 3] = x[b]; ep[o + 4] = y[b]; ep[o + 5] = z[b];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.edgePosBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, ep);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.depthMask(false);
      const E = this.edge;
      gl.useProgram(E.p);
      gl.uniformMatrix4fv(E.u.uViewProj, false, cam.viewProj);
      gl.uniform3fv(E.u.uEye, cam.eye);
      gl.uniform2f(E.u.uFog, cam.fogNear, cam.fogFar);
      gl.uniform1f(E.u.uAlpha, opts.edgeAlpha);
      gl.uniform3fv(E.u.uEdgeColor, opts.edgeColor);
      gl.bindVertexArray(this.edgeVao);
      gl.drawArrays(gl.LINES, 0, this.edgeCount * 2);
      gl.depthMask(true);
    }
    gl.bindVertexArray(null);
  }
}
