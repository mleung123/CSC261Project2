use std::collections::HashMap;
use rand::prelude::*;
use rand::distr::OpenClosed01;

const MAX_EXCHANGE_PAIRS: i32=20; // i.e. each agent makes MAX_EXCHANGE_PAIRS offers.
const NUM_RESOURCE_TYPES: u32 = 2;
const MAX_RESOURCES: u32 = 4;
const MAX_RESOURCES_INC: u32 = MAX_RESOURCES + 1;
const MAX_FAILURES: u32 = 5;

const PRINTING: bool = false;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum NegotiationMessage{
    Accept,
    Offer(Vec<u32>),
    Empty
}

impl NegotiationMessage {
    fn create_random(rand: &mut ThreadRng) -> NegotiationMessage {
        let c: f64= rand.sample::<f64, OpenClosed01>(OpenClosed01)*(MAX_RESOURCES.pow(2)) as f64+1.0;
        if c>MAX_RESOURCES.pow(2) as f64 {
            return NegotiationMessage::Accept;
        }
        NegotiationMessage::Offer(vec![rand.random_range(0..MAX_RESOURCES+1), rand.random_range(0..MAX_RESOURCES+1)])
    }

    fn invert(&self) -> NegotiationMessage {
        if let NegotiationMessage::Offer(vec) = self {
            let response = vec![ MAX_RESOURCES - vec[0], MAX_RESOURCES - vec[1] ];
            return NegotiationMessage::Offer(response);
        } else {
            return self.clone()
        }
    }
}

pub trait RL {
    fn send(&mut self, exploration_rate: f32, n: NegotiationMessage) -> NegotiationMessage;
    fn compute_reward_and_update_q(&mut self, final_offer: &NegotiationMessage);
}

fn init_q_table_entry() -> HashMap<NegotiationMessage, f32> {
    let mut map = HashMap::with_capacity(MAX_RESOURCES_INC.pow(NUM_RESOURCE_TYPES) as usize + 1);
    for i in 0..MAX_RESOURCES_INC {
        for j in 0..MAX_RESOURCES_INC {
            map.insert(NegotiationMessage::Offer(vec![ i, j ]), 0.0);
        }
    }
    map.insert(NegotiationMessage::Accept, 0.0);
    return map;
}

struct QLearning {
    q_table: HashMap<(NegotiationMessage, u32), HashMap<NegotiationMessage, f32>>,
    offer_count: HashMap<NegotiationMessage, u32>, // number of times each offer has been sent within an episode
    learning_rate: f32,
    gamma: f32,
    rng: ThreadRng,
    reward_table: Vec<i32>,
    episode_history: Vec<((NegotiationMessage, u32), NegotiationMessage)>
}

impl QLearning {
    fn new(learning_rate: f32, gamma: f32, reward_table: Vec<i32>) -> Self {

        // Calc num states and actions
        let actions = (MAX_RESOURCES + 1).pow(NUM_RESOURCE_TYPES) as usize + 1;
        let states = (MAX_RESOURCES + 1).pow(NUM_RESOURCE_TYPES) as usize * MAX_FAILURES as usize;
        let capacity = (actions * states) as usize;
        println!("Capacity: {capacity}, states: {states}, actions: {actions}");
        let q_table = HashMap::with_capacity(capacity);

        Self {
            q_table,
            offer_count: HashMap::new(),
            learning_rate,
            gamma,
            rng: rand::rng(),
            reward_table,
            episode_history: Vec::with_capacity(MAX_EXCHANGE_PAIRS as usize*2)
        }
    }

    fn increment_offer_count(&mut self, message: &NegotiationMessage) {
        if self.offer_count.contains_key(message){
            let x = self.offer_count.get_mut(message);
            if let Some(val) = x {
                *val+=1
            }
        }
        else{
            self.offer_count.insert(message.clone(), 1);
        }
    }

    fn get_max_offer_for_state(&mut self, state: (NegotiationMessage, u32)) -> (NegotiationMessage, f32) {
        // Get or init
        let mut action_weights = self.q_table.entry(state).or_insert_with(init_q_table_entry).iter();
        // Use first variable as max
        let mut all_weights_equal = true;
        let (mut max_action, mut max_weight) = action_weights.next().expect("Weights should have been initialzed");

        // Find the max action
        for (action, weight) in action_weights {
            if weight > max_weight {
                max_action = action;
                max_weight = weight;
            } else if weight != max_weight {
                all_weights_equal = false;
            }
        }

        // Return random value if all weights are equal (if we don't do this we will always return the first msg)
        if all_weights_equal {
            (NegotiationMessage::create_random(&mut self.rng), *max_weight)
        } else {
            (max_action.clone(), *max_weight)
        }
    }
}



impl RL for QLearning {
    fn send(&mut self, exploration_rate: f32, message: NegotiationMessage) -> NegotiationMessage {
        let mut rng = rand::rng();
        rng.reseed();
        
        if exploration_rate<rng.sample::<f32, OpenClosed01>(OpenClosed01){
            
            rng.reseed();
            let mut reply = NegotiationMessage::create_random(&mut rng);
            self.increment_offer_count(&reply);
            // println!("returning reply");
            return reply;   
        }
        
        let current_state = (message.clone(), self.offer_count.get(&message).copied().unwrap_or(0));
        let (reply, _) = self.get_max_offer_for_state(current_state.clone());

        self.episode_history.push((current_state, reply.clone()));
        self.increment_offer_count(&reply);
        
        return reply; // Return the chosen action
    }

    // TODO what does sender get?
    fn compute_reward_and_update_q(&mut self, final_offer: &NegotiationMessage) {
        let mut reward: i32 = 0;
        match final_offer {
            NegotiationMessage::Empty =>  {}, // max number of messages
            NegotiationMessage::Accept => eprintln!("final offer was Accept!"), // case never happens
            NegotiationMessage::Offer(offer) => {
                offer.iter().enumerate().for_each(|(i, val)| reward+=*val as i32*self.reward_table[i]) 
            }
        }
        reward -= self.offer_count.values().sum::<u32>() as i32 *10;

        // update q-table
        let reward_f = reward as f32;
        println!("An agent was rewarded by {reward}");
        for (state, action) in self.episode_history.clone() {
            let (_, max_weight) = self.get_max_offer_for_state(state.clone());
            let action_map = self.q_table.entry(state).or_insert_with(init_q_table_entry);
            let current_q = *action_map.get(&action).unwrap_or(&0.0); // 0 default

            let new_q = current_q + self.learning_rate * (reward_f + self.gamma * max_weight);

            action_map.insert(action.clone(), new_q);
        }
        
        self.episode_history.clear(); 
        self.offer_count.clear();
    }
}

fn main() {
    println!("Hello, world!");

    let mut agent_1 = QLearning::new(0.1, 0.9, vec![10, 5]);
    let mut agent_2 = QLearning::new(0.1, 0.9, vec![5, 10]);

    let explore_rates =[0.95, 0.8,0.5,0.3,0.1];
    //let n_episodes=[100,100,100,100,100];
    let n_episodes=[2500,2500,2500,2500,2500];
    for i in 0..explore_rates.len(){
        epoch_driver(&mut agent_1, &mut agent_2,explore_rates[i],n_episodes[i]);
    }
    
}

fn episode_driver<T: RL>(mut  agent_1: &mut T, mut agent_2: &mut T, exploration_rate: f32, ep_num: u32) {

    println!("\nstarting episode {}", ep_num);
    
    
    let mut messages: Vec<NegotiationMessage> = Vec::new();

    let mut current_message: NegotiationMessage = NegotiationMessage::Empty; 
    
    let mut num_rounds: i32 = 0;
    

    while current_message != NegotiationMessage::Accept && num_rounds < MAX_EXCHANGE_PAIRS {

        let agent_1_offer: NegotiationMessage = agent_1.send(exploration_rate,current_message);
        messages.push(agent_1_offer.clone());
        // println!("Round {}: Agent 1 sends: {:?}", num_rounds, agent_1_offer);

        if agent_1_offer == NegotiationMessage::Accept { 
            current_message = agent_1_offer;
            break; 
        }

        let agent_2_offer: NegotiationMessage = agent_2.send(exploration_rate,agent_1_offer);
        messages.push(agent_2_offer.clone());
        // println!("Round {}: Agent 2 sends: {:?}", num_rounds, agent_2_offer);
        
        current_message = agent_2_offer;
        
        if current_message == NegotiationMessage::Accept { break; }

        num_rounds += 1;
    }

    let mut final_outcome = messages.last().cloned().unwrap_or(NegotiationMessage::Empty);
    if final_outcome == NegotiationMessage::Accept {
        if messages.len() > 1 {
            final_outcome = messages[messages.len() - 2].clone()
        } else {
            final_outcome = NegotiationMessage::Empty
        }
    }

    println!("outcome: {:?}", final_outcome);
    println!("rounds: {}", num_rounds);

    if messages.len() % 2 == 0 { // agent 2 had las offer
        agent_1.compute_reward_and_update_q(&final_outcome.invert());
        agent_2.compute_reward_and_update_q(&final_outcome);
    } else {
        agent_1.compute_reward_and_update_q(&final_outcome);
        agent_2.compute_reward_and_update_q(&final_outcome.invert());
    }
    
}

fn epoch_driver<T: RL>(mut agent_1: &mut T, mut agent_2: &mut T, exploration_rate: f32, n_episodes: u32) {
    let mut ep_num = 0;

    for _i in 0..n_episodes{
        episode_driver(agent_1, agent_2,  exploration_rate,ep_num);
        ep_num+=1;
    }
}




//PHC notes:

//phc essentially comes down to replacing the greedy policy with a new mixed policy, and updating the mixed policy at each step.


// to implement this, we'll need to have a policy table keeping track of the likelihood of choosing any given state-action pair.
// we could either make a separate table for this, or make it so the q-table has a tuple containing the q-value and this probability.


// mixed policy:
// pick a random possible action, using the probabilities in the policy table.

//updating policy:

//for every single sa-pair:

    //  we increase the likelihood of choosing 
    //  the highest valued action by the combined delta_a of every single other sa-pair.
    //  and decrease the likelihood of choosing

// where delta_a is the minimum of the current likelihood of choosing the action, 
// or delta(some constant) divided by number of other possible actions.

//note: I have no idea how we're supposed to ensure that the probabilities sum to one at any given state
// we could just completely brush this under the rug...

// not sure if it makes sense to create a new struct for this, since it's very similar to q-learning. 
// At the same time, it does need a policy table, so it just cant't be a trait.. 
