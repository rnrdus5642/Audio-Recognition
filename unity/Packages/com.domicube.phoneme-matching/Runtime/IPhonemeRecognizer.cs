using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    /// <summary>
    /// Audio -> IPA phonemes. The matching layer consumes only the
    /// phoneme list and does not care which model produced it, so the
    /// acoustic model can land later (ONNX export is a separate phase)
    /// without the rest of the package waiting on it.
    ///
    /// Implementations must emit phonemes from the same inventory the
    /// targets were built with - for Korean that means running the
    /// recogniser's text output through
    /// <see cref="Korean.JamoIpa.ToPhonemes"/>.
    /// </summary>
    public interface IPhonemeRecognizer
    {
        /// <summary>ISO language code, e.g. "ko".</summary>
        string Language { get; }

        /// <summary>
        /// Recognise 16 kHz mono float samples in roughly [-1, 1].
        /// Returns an empty list when no speech was detected.
        /// </summary>
        List<string> Recognize(float[] audio16kMono);

        /// <summary>
        /// Raw transcript alongside the phonemes, from a single pass.
        ///
        /// Models that transcribe first and then apply G2P would run the
        /// acoustic model twice if the caller asked for each separately -
        /// which matters when every frame of a live stream is scored.
        /// Implementations with no separate transcript may return an
        /// empty string.
        /// </summary>
        void RecognizeWithText(
            float[] audio16kMono, out string text, out List<string> phonemes);
    }
}
