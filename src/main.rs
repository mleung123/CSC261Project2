use std::collections::HashMap;
use rand::prelude::*;
use rand::distr::OpenClosed01;


const NUM_RESOURCE_TYPES: u32 = 2;
const MAX_RESOURCES: u32 = 4;
const MAX_FAILURES: u32 = 5;

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum NegotiationMessage{
    Accept,
    Offer(Vec<u32>), // Should we use [u32; NUM_RESOURCE_TYPES]?
    Empty
}

pub trait RL {
    fn send(&mut self, n: NegotiationMessage) -> NegotiationMessage;
    fn compute_reward_and_update_q(&mut self, personal_dialog: &Vec<NegotiationMessage>);
}

struct QLearning {
    q_table: HashMap<(NegotiationMessage, u32), f32>,
    offer_count: HashMap<NegotiationMessage, u32>, // number of times each offer has been sent within an episode
    learning_rate: f32,
    gamma: f32,
    exploration_rate: f32
}

impl QLearning {
    fn new(learning_rate: f32, gamma: f32, exploration_rate: f32) -> Self {
        
        
        
        let actions = (MAX_RESOURCES + 1).pow(NUM_RESOURCE_TYPES) as usize + 1;
        let states = (MAX_RESOURCES + 1).pow(NUM_RESOURCE_TYPES) as usize * MAX_FAILURES as usize;
        let capacity = (actions * states) as usize;
        println!("Capacity: {capacity}, states: {states}, actions: {actions}");
        let mut q_table = HashMap::with_capacity(capacity);
        
        for i in 0..MAX_FAILURES {
            // 2 loops, 2 NUM_RESOURCE_TYPES
            for j in 0..(MAX_RESOURCES + 1) {
                for k in 0..(MAX_RESOURCES + 1) {
                    let message = NegotiationMessage::Offer(vec![ j, k ]);
                    q_table.insert((message, i), 0.0);
                }
            }
            q_table.insert((NegotiationMessage::Accept, i), 0.0);
        }

        Self { q_table, offer_count:HashMap::new(), learning_rate, gamma, exploration_rate }
    }
}

impl RL for QLearning {
    fn send(&mut self, message: NegotiationMessage) -> NegotiationMessage {
        let mut rng = rand::rng();
        
        // Explore case
        if self.exploration_rate<rng.sample::<f32, OpenClosed01>(OpenClosed01){
            let a: f64= rng.sample::<f64, OpenClosed01>(OpenClosed01)*MAX_RESOURCES as f64;
            let b: f64= rng.sample::<f64, OpenClosed01>(OpenClosed01)*MAX_RESOURCES as f64;
            
            let mut reply = NegotiationMessage::Offer(vec![ a.round() as u32, b.round() as u32 ]);
            

            //if this isn't the first message, then accept is a valid random action.
            if matches!(message, NegotiationMessage::Empty){
                let c: f64= rng.sample::<f64, OpenClosed01>(OpenClosed01)*(MAX_RESOURCES.pow(2)) as f64+1.0;
                if c>MAX_RESOURCES.pow(2) as f64 {
                    reply =NegotiationMessage::Accept;
                }
            }
            else{ //update offer_count
                if self.offer_count.contains_key(&reply){
                    let x = self.offer_count.get_mut(&reply);
                    if let Some(val) = x {
                        *val+=1
                    }
                }
                else{
                    self.offer_count.insert(reply.clone(),1);
                }
            }
            return reply;   
        }
        let mut max = 0.0;
        let mut max_message = NegotiationMessage::Offer(vec![]);
        
        //find highest-valued valid action.
        for ((message,count),val) in self.q_table.iter(){
            if matches!(self.offer_count.get(message), Some(count)){
                if max<*val {
                    max = *val;
                    max_message=message.clone();
                }
            }
        }
        //todo: turn this into a function
        if self.offer_count.contains_key(&max_message){
            let x = self.offer_count.get_mut(&max_message);
                if let Some(val) = x {
                    *val+=1
                }
        }
        else{
            self.offer_count.insert(max_message.clone(),1);
        }
        return max_message;

        
    }

    fn compute_reward_and_update_q(&mut self, personal_dialog: &Vec<NegotiationMessage>) {
        todo!()
    }
}

fn main() {
    println!("Hello, world!");

    let agent_1 = QLearning::new(0.1, 0.9, 0.95);
    let agent_2 = QLearning::new(0.1, 0.9, 0.95);
    episode_driver(agent_1, agent_2, 0);
}

fn episode_driver(mut agent_1: impl RL, mut agent_2: impl RL, ep_num: u32) {

    // Episode
    let mut messages: Vec<NegotiationMessage> = Vec::new();
    // TODO make more reasonable default
    let mut last_message: NegotiationMessage = NegotiationMessage::Offer(vec![]);
    
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
