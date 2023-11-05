use std::{fmt::Debug, ops::Mul};

pub struct BeamSearch<L> {
    pub sequences: Vec<L>,
    pub beam_size: usize,
}

impl<L> BeamSearch<L> {
    pub fn new(beam_size: usize) -> Self {
        BeamSearch {
            sequences: Vec::new(),
            beam_size,
        }
    }

    pub fn add_sequence(&mut self, sequence: L) {
        self.sequences.push(sequence);
    }
}

impl<L: Iterator + Clone> BeamSearch<L>
where
    for<'a> &'a <L as Iterator>::Item: Mul<f64>,
    for<'a> <&'a <L as Iterator>::Item as Mul<f64>>::Output: TryInto<f64>,
    for<'a> <<&'a <L as Iterator>::Item as Mul<f64>>::Output as TryInto<f64>>::Error: Debug,
{
    /// Select the sequence with the highest conditonal probability, taken from all tokens.
    ///
    /// `<L as Iterator>::Item` must implement `Mul`, whose `::Output` must implement `TryInto<f64>`,
    /// and whose `::Error` must implement `Debug`.
    pub fn select_sequence(self) -> Vec<L> {
        let mut max = f64::MIN;
        let mut max_seq = None;

        for seq in self.sequences {
            let cond_prob = seq
                .clone()
                .fold(1.0, |acc, ref x| (x * acc).try_into().unwrap());

            if cond_prob >= max {
                match max_seq {
                    None => {
                        max_seq = Some(vec![seq]);
                    }
                    Some(ref mut seqs) => {
                        seqs.push(seq);
                    }
                }
                max = cond_prob;
            }
        }

        max_seq.unwrap_or(vec![])
    }
}
