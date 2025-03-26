use std::{collections::HashMap};

#[derive(PartialEq, Eq, Hash)]
pub enum NegotiationMessage{
    Accept,
    Reject,
    Offer(Vec<u32>)
}

pub trait RL {
    fn send(&mut self, n: NegotiationMessage) -> NegotiationMessage;
    fn compute_reward(personal_dialog: &Vec<NegotiationMessage>);
}

struct QLearning {
    QTable: HashMap<(NegotiationMessage, u32), f32>,
    learning_rate: f32,
    gamma: f32,
    exploration_rate: f32
}

fn main() {
    println!("Hello, world!");
}
