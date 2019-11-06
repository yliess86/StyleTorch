uniform sampler2D texture;
varying vec2      v_texcoord;

void main()
{
    vec3 color   = texture2D( texture, v_texcoord ).rgb;
    gl_FragColor = vec4( color, 1. );
}
