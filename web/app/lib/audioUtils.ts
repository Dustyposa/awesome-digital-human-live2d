/**
 * Converts a Float32Array to an Int16Array.
 * The input buffer is assumed to contain PCM audio data ranging from -1.0 to 1.0.
 * @param buffer The Float32Array to convert.
 * @returns A new Int16Array.
 */
export const float32ToInt16 = (buffer: Float32Array): Int16Array => {
  const int16Array = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    const val = Math.max(-1, Math.min(1, buffer[i])); // Clamp to [-1, 1]
    int16Array[i] = val < 0 ? val * 0x8000 : val * 0x7FFF; // Convert to 16-bit signed int
  }
  return int16Array;
};

/**
 * Converts an Int16Array to a base64 encoded string.
 * This is useful for sending 16-bit PCM audio data over WebSockets.
 * @param int16Array The Int16Array to convert.
 * @returns A base64 encoded string.
 */
export const int16ArrayToBase64 = (int16Array: Int16Array): string => {
  // Int16Array.buffer gives an ArrayBuffer.
  // We need to create a Uint8Array view on this buffer to access individual bytes.
  const uint8Array = new Uint8Array(int16Array.buffer);
  let binaryString = '';
  for (let i = 0; i < uint8Array.byteLength; i++) {
    binaryString += String.fromCharCode(uint8Array[i]);
  }
  return btoa(binaryString);
};

/**
 * Resamples an audio buffer from a source sample rate to a target sample rate.
 * This is a basic implementation (linear interpolation) and might not be suitable for high-quality audio.
 * For better quality, a more sophisticated algorithm (e.g., using sinc functions) or a library is recommended.
 * 
 * @param buffer The input audio buffer (Float32Array).
 * @param sourceSampleRate The sample rate of the input buffer.
 * @param targetSampleRate The desired output sample rate.
 * @returns A new Float32Array containing the resampled audio.
 */
export const resampleBuffer = (
  buffer: Float32Array,
  sourceSampleRate: number,
  targetSampleRate: number
): Float32Array => {
  if (sourceSampleRate === targetSampleRate) {
    return buffer; // No resampling needed
  }

  const ratio = targetSampleRate / sourceSampleRate;
  const outputLength = Math.round(buffer.length * ratio);
  const outputBuffer = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const t = i / ratio; // Corresponding time in the source buffer
    const index1 = Math.floor(t);
    const index2 = Math.min(Math.ceil(t), buffer.length - 1); // Ensure index2 is within bounds
    const fraction = t - index1;

    if (index1 === index2 || index1 >= buffer.length -1) { // Edge case: if index1 is the last sample or indices are same
        outputBuffer[i] = buffer[index1];
    } else {
        // Linear interpolation
        outputBuffer[i] = buffer[index1] * (1 - fraction) + buffer[index2] * fraction;
    }
  }
  return outputBuffer;
};

// Example usage (for testing):
/*
const exampleFloat32 = new Float32Array([-1, -0.5, 0, 0.5, 1]);
const exampleInt16 = float32ToInt16(exampleFloat32);
console.log(exampleInt16); // Expected: Int16Array [-32768, -16384, 0, 16383, 32767]
const exampleBase64 = int16ArrayToBase64(exampleInt16);
console.log(exampleBase64);
*/

/*
// Resampling test
const testRate = 48000;
const targetRate = 16000;
const testDuration = 1; // 1 second
const testBuffer = new Float32Array(testRate * testDuration);
// fill with a simple sine wave
for (let i = 0; i < testBuffer.length; i++) {
    testBuffer[i] = Math.sin(2 * Math.PI * 440 * (i / testRate));
}
console.log(`Original buffer length (${testRate}Hz): ${testBuffer.length}`);
const resampled = resampleBuffer(testBuffer, testRate, targetRate);
console.log(`Resampled buffer length (${targetRate}Hz): ${resampled.length}`); // Expected: testBuffer.length / 3
*/
