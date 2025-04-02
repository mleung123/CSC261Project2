use std::{collections::HashMap, ops::Neg};

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum NegotiationMessage{
    Accept,
    Reject,
    Offer(Vec<u32>)
}

pub trait RL {
    fn send(&mut self, n: NegotiationMessage) -> NegotiationMessage;
    fn compute_reward(&mut self, personal_dialog: &Vec<NegotiationMessage>);
}

struct QLearning {
    QTable: HashMap<(NegotiationMessage, u32), f32>,
    learning_rate: f32,
    gamma: f32,
    exploration_rate: f32
}

fn main() {
    println!("Hello, world!");

    driver();
}

fn driver(mut agent_1: impl RL, mut agent_2: impl RL, ep_num: u32) {

    // Episode
    let mut messages: Vec<NegotiationMessage> = Vec::new();
    let mut last_message: NegotiationMessage = NegotiationMessage::Reject;
    
    while last_message != NegotiationMessage::Accept {

        // Send message to agent 1
        let offer: NegotiationMessage = agent_1.send(last_message);
        messages.push(offer.clone());

        // Check if offer is the end of the negotiation episode
        if offer == NegotiationMessage::Accept { break }

        last_message = agent_2.send(offer);
        messages.push(last_message.clone());
    }
}
