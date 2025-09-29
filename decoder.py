import numpy as np
from collections import defaultdict

class CTCBeamSearchDecoder:
    def __init__(self, idx2char, beam_width=10, blank=0):
       
        self.idx2char = idx2char
        self.beam_width = beam_width
        self.blank = blank

    def decode(self, log_probs):

        if log_probs.dim() == 3 and log_probs.shape[0] != log_probs.shape[1]:
            # assume (T, B, C)
            log_probs = log_probs.permute(1, 0, 2)  # -> (B, T, C)

        batch_size = log_probs.size(0)
        results = []
        for b in range(batch_size):
            results.append(self._decode_single(log_probs[b]))
        return results

    def _decode_single(self, log_probs):
       
        time_steps, vocab_size = log_probs.size()
        log_probs = log_probs.cpu().detach().numpy()

        # Beam: seq -> (log_prob_blank, log_prob_non_blank)
        beam = {(): (0.0, -np.inf)}

        for t in range(time_steps):
            next_beam = defaultdict(lambda: (-np.inf, -np.inf))
            for seq, (p_b, p_nb) in beam.items():
                for c in range(vocab_size):
                    p = log_probs[t, c]

                    if c == self.blank:
                        # Extend with blank: stays same seq
                        nb = next_beam[seq]
                        next_beam[seq] = (
                            np.logaddexp(nb[0], p_b + p),
                            np.logaddexp(nb[1], p_nb + p)
                        )
                    else:
                        new_seq = seq + (c,)
                        if len(seq) > 0 and seq[-1] == c:
                            # Extend same char: only from blank
                            nb = next_beam[new_seq]
                            next_beam[new_seq] = (
                                nb[0],
                                np.logaddexp(nb[1], p_b + p)
                            )
                        else:
                            # Extend with new char
                            nb = next_beam[new_seq]
                            next_beam[new_seq] = (
                                nb[0],
                                np.logaddexp(nb[1], max(p_b, p_nb) + p)
                            )

            # Prune beams
            beam = dict(sorted(
                next_beam.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )[:self.beam_width])

        # Pick best sequence
        best_seq, (p_b, p_nb) = max(
            beam.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1])
        )

        # Collapse repeats & remove blanks
        decoded = []
        prev = self.blank
        for c in best_seq:
            if c != prev and c != self.blank:
                if c in self.idx2char:       
                    decoded.append(self.idx2char[c])
            prev = c

        return "".join(decoded)
