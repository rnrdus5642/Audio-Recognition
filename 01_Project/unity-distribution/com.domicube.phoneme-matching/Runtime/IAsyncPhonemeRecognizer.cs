using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace DomiCube.PhonemeMatching
{
    /// <summary>A transcript and its phonemes from the same completed inference.</summary>
    public sealed class RecognitionResult
    {
        public string Text { get; }
        public List<string> Phonemes { get; }

        public RecognitionResult(string text, List<string> phonemes)
        {
            Text = text ?? string.Empty;
            Phonemes = phonemes ?? new List<string>();
        }
    }

    /// <summary>
    /// Optional non-blocking path. Call on the engine thread. One inference may
    /// own the backend at a time; cancellation retires its result but the backend
    /// may stay busy while submitted GPU work drains. Never reuse its tensors
    /// until IsRecognizing is false. A pending task is not an empty transcript.
    /// </summary>
    public interface IAsyncPhonemeRecognizer : IPhonemeRecognizer
    {
        bool IsRecognizing { get; }
        Task<RecognitionResult> RecognizeAsync(float[] audio16kMono, CancellationToken cancellation);
    }
}
