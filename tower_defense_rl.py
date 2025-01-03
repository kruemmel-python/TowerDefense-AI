import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LayerNormalization, Lambda
from tensorflow.keras.optimizers import Adam
import os
import random
import time
from collections import deque
import math
import matplotlib.pyplot as plt

# Konfiguration
WIDTH, HEIGHT = 700, 700
GRID_SIZE = 10  # Spielfeldgröße
CELL_SIZE = WIDTH // GRID_SIZE  # Größe der einzelnen Zellen
NUM_EPISODES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_RATE = 0.0005
TARGET_UPDATE_FREQ = 100
MODEL_PATH = "tower_defense_agent.keras"
RENDER_EVERY = 10
PRIORITIZED_REPLAY_EPS = 1e-6
MEMORY_SIZE_START = 1000
MEMORY_SIZE_MAX = 20000

# Tower und Enemy Typen
TOWER_TYPES = ["basic", "laser", "slow", "area", "energy"]  # Hinzugefügt: Area-Schaden, Energie-Schuss
ENEMY_TYPES = ["basic", "fast", "strong", "armored"]  # Hinzugefügt: Gepanzert
NUM_TOWER_TYPES = len(TOWER_TYPES)
NUM_ENEMY_TYPES = len(ENEMY_TYPES)


# Aktionen (Indexbasiert)
ACTIONS = [
    "do_nothing",
    "place_tower_basic",
    "place_tower_laser",
    "place_tower_slow",
    "place_tower_area",   # Hinzugefügt
    "place_tower_energy",  # Hinzugefügt
    "upgrade_tower",
    "use_ability"
]

NUM_ACTIONS = len(ACTIONS)

# Belohnungen
REWARD_ENEMY_KILL = 1
REWARD_WAVE_COMPLETE = 10
REWARD_BASE_SURVIVAL = 0.1
PENALTY_ENEMY_HIT_BASE = -10
PENALTY_WASTED_RESOURCES = -0.1
REWARD_STEP = -0.005


class TowerDefenseEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tower Defense")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) # 0 = leer, 1= tower
        self.towers = []  # Liste von Türmen (Typ, x, y, level, hp)
        self.enemies = []  # Liste von Gegner (Typ, x, y, hp, speed)
        self.resources = 150  # Startressourcen
        self.wave = 1
        self.base_hp = 100
        self.done = False
        self.enemy_spawn_timer = 0
        self.enemy_spawn_delay = 50
        return self.get_state()

    def get_state(self):
        state = []

        # Normalisierte Ressourcen
        state.append(self.resources / 200)  # Normalize by max resources
        state.append(self.base_hp / 100) # Normalize by max base hp

        # Aktuelle Welle
        state.append(self.wave / 10)  # Normalize max wave

        # Grid Informationen (ob ein Turm existiert)
        for row in self.grid:
            for cell in row:
                state.append(cell) # 0 falls kein Turm, 1 falls Turm
        
        # Pad tower and enemy information to fixed sizes
        MAX_TOWERS = 50
        MAX_ENEMIES = 50

        # Normalisierte Turm Informationen
        for tower in self.towers:
           state.append(tower[1] / WIDTH)
           state.append(tower[2] / HEIGHT) # X,Y Position
           state.append(TOWER_TYPES.index(tower[0]) / (len(TOWER_TYPES) -1)) # Typ des Turms (normalisiert)
           state.append(tower[3] / 5 ) # Level des Turms (max Level 5)
           state.append(tower[4] / 100) # Turm HP
        
        # Pad with zeros if there are not enough towers
        for _ in range(max(0, MAX_TOWERS - len(self.towers))):
          state.extend([0,0,0,0,0])


        # Normalisierte Gegner-Informationen
        for enemy in self.enemies:
             state.append(enemy[1] / WIDTH)
             state.append(enemy[2] / HEIGHT) # X,Y Position
             state.append(ENEMY_TYPES.index(enemy[0]) / (len(ENEMY_TYPES) - 1)) # Typ
             state.append(enemy[3] / 100)  # Normalized health
             state.append(enemy[4] / 5)    # Normalized speed

        # Pad with zeros if there are not enough enemies
        for _ in range(max(0, MAX_ENEMIES - len(self.enemies))):
           state.extend([0,0,0,0,0])
        
        return np.array(state, dtype=np.float32)
      
    def calculate_state_size(self):
      state_size = 2 # resources, base hp
      state_size += 1 # wave
      state_size += GRID_SIZE * GRID_SIZE  # Grid Informationen
      state_size += 50 * 5 # fixed size of 50 towers
      state_size += 50 * 5 # fixed size of 50 enemies

      return state_size


    def spawn_enemies(self):
      # Spawnt eine neue Welle mit immer mehr Gegner
      if len(self.enemies) == 0:
           self.enemy_spawn_timer += 1
           if self.enemy_spawn_timer >= self.enemy_spawn_delay:
            num_enemies = 2 + self.wave # Increase enemy count per wave
            for _ in range(num_enemies):
                enemy_type = random.choice(ENEMY_TYPES)
                enemy_x = 0
                enemy_y = random.randint(0, HEIGHT - 10)  # Random y position
                enemy_hp = 20 + (self.wave * 5)  # Increase health per wave

                # Set enemy speed
                enemy_speed = 2
                if enemy_type == "fast":
                   enemy_speed = 4 # fast enemies move faster
                
                self.enemies.append([enemy_type, enemy_x, enemy_y, enemy_hp, enemy_speed])
                print(f"Enemy spawned with HP: {enemy_hp}") # Debugging

            self.enemy_spawn_timer = 0
            self.wave += 1


    def move_enemies(self):
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            print(f"Enemy HP before step: {enemy[3]}") # Debugging
            enemy[1] += enemy[4]  # Move enemy based on its speed
            if enemy[1] > WIDTH:  # Enemy has hit the base
                self.enemies.pop(i)
                self.base_hp -= 20
                if self.base_hp <= 0:
                    self.done = True
            else:  # Check if enemy is close to a tower
                for tower_idx, tower in enumerate(self.towers):
                    distance = np.sqrt((enemy[1] - tower[1]) ** 2 + (enemy[2] - tower[2]) ** 2)
                    if distance <= 20:  # Enemy is close to tower
                      enemy_damage = 5 + (self.wave * 2)
                      if enemy[0] == "strong":
                        enemy_damage = 10 + (self.wave * 3)
                      elif enemy[0] == "armored":
                        enemy_damage = 3 + (self.wave * 1)

                      tower[4] -= enemy_damage
                      
                      if tower[4] <= 0: # Remove tower if health is below 0
                          tower_x = tower[1] // CELL_SIZE
                          tower_y = tower[2] // CELL_SIZE
                          self.grid[tower_y][tower_x] = 0 # Mark as empty
                          self.towers.pop(tower_idx) # Remove Tower

        # Remove enemies which hit the base
        for enemy_idx in reversed(enemies_to_remove):
            if enemy_idx < len(self.enemies):
                self.enemies.pop(enemy_idx)


    def place_tower(self, tower_type, x, y):
        grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
        if self.resources >= 20 and self.grid[grid_y][grid_x] == 0:
            self.grid[grid_y][grid_x] = 1 # Mark as tower here
            tower_hp = 100 # Anfangs-HP des Turms
            self.towers.append([tower_type, x, y, 1, tower_hp]) # Typ, x,y,level, hp
            self.resources -= 20
            return True # Successful placement
        return False

    def upgrade_tower(self, x, y):
        grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
        
        # Suche den Turm am entsprechenden Ort
        for tower in self.towers:
          if (tower[1] // CELL_SIZE == grid_x and tower[2] // CELL_SIZE == grid_y):
            if tower[3] < 5 and self.resources >= 15:
              tower[3] +=5 # Upgrade Tower Level
              self.resources -= 15
              return True
        return False

    def use_ability(self):
      if self.resources >= 10:
        self.resources -=10
        for enemy in self.enemies:
          enemy[3] -= 10
        return True
      return False

    def handle_tower_attacks(self):
        for tower in self.towers:
            tower_type, x, y, level, _ = tower
            
            targets_in_range = []
            for enemy in self.enemies:
                enemy_x, enemy_y = enemy[1], enemy[2]
                distance = np.sqrt((x - enemy_x)**2 + (y - enemy_y)**2)

                # Define range of each tower
                tower_range = 100 # Basic Tower
                if tower_type == "laser":
                    tower_range = 150
                elif tower_type == "slow":
                    tower_range = 80
                elif tower_type == "area": # AOE Tower
                  tower_range = 70
                elif tower_type == "energy": # Energy tower
                  tower_range = 120

                if distance <= tower_range:
                  targets_in_range.append(enemy)

            if targets_in_range:
                target = self.get_closest(tower, targets_in_range) # Get closest

                damage = 5 + (level * 2)

                if tower_type == "area":
                     for enemy in targets_in_range:
                         enemy[3] -= damage/2 # Area towers do half damage to all in range
                elif target: # All other towers deal damage to closest target
                  if tower_type == "energy":
                    target[3] -= damage * 2  # Energy tower does more damage
                  else:
                    target[3] -= damage # All others use normal damage

                if tower_type == "slow":
                     for enemy in targets_in_range:
                      enemy[1] -= 1  # slow down by 1 pixel (add effect)
    

    def get_closest(self, tower, targets):
        if not targets:
            return None

        closest_target = None
        min_distance = float('inf')

        for target in targets:
            distance = np.sqrt((tower[1] - target[1])**2 + (tower[2] - target[2])**2)
            if distance < min_distance:
                min_distance = distance
                closest_target = target
        return closest_target

    def enemies_attack_towers(self):
        for enemy in self.enemies:
            enemy_x = enemy[1]
            enemy_y = enemy[2]
            
            # Prüfe, ob ein Turm in Reichweite ist
            for tower in self.towers:
                tower_x = tower[1]
                tower_y = tower[2]
                distance = np.sqrt((tower_x - enemy_x)**2 + (tower_y - enemy_y)**2)
                
                if distance <= 50:  # Angriffsreichweite des Gegners
                    tower[4] -= 10  # Gegner fügt dem Turm 10 Schaden zu
                    
                    # Entferne den Turm, falls seine HP <= 0 sind
                    if tower[4] <= 0:
                        self.towers.remove(tower)
                        self.grid[tower_y // CELL_SIZE][tower_x // CELL_SIZE] = 0  # Entferne Turm vom Grid


    def step(self, action_index):
        reward = REWARD_STEP
        original_resources = self.resources
        
        if action_index > 0:
          action_name = ACTIONS[action_index]
        else:
          action_name = ACTIONS[0] # do_nothing

        if action_name == "place_tower_basic":
          placement_successful = False
          for i in range(100): # Try 100 times
            random_x = random.randint(0, WIDTH - 1)
            random_y = random.randint(0, HEIGHT - 1)
            placement_successful = self.place_tower("basic", random_x, random_y)
            if placement_successful:
              break;
        elif action_name == "place_tower_laser":
          placement_successful = False
          for i in range(100):
            random_x = random.randint(0, WIDTH - 1)
            random_y = random.randint(0, HEIGHT - 1)
            placement_successful = self.place_tower("laser", random_x, random_y)
            if placement_successful:
                break
        elif action_name == "place_tower_slow":
          placement_successful = False
          for i in range(100):
            random_x = random.randint(0, WIDTH - 1)
            random_y = random.randint(0, HEIGHT - 1)
            placement_successful = self.place_tower("slow", random_x, random_y)
            if placement_successful:
                break
        elif action_name == "place_tower_area": # Place AOE
          placement_successful = False
          for i in range(100):
            random_x = random.randint(0, WIDTH - 1)
            random_y = random.randint(0, HEIGHT - 1)
            placement_successful = self.place_tower("area", random_x, random_y)
            if placement_successful:
                break
        elif action_name == "place_tower_energy": # Place Energy
          placement_successful = False
          for i in range(100):
            random_x = random.randint(0, WIDTH - 1)
            random_y = random.randint(0, HEIGHT - 1)
            placement_successful = self.place_tower("energy", random_x, random_y)
            if placement_successful:
                break
        elif action_name == "upgrade_tower":
            if self.towers: # Upgrade only if towers exists
               random_tower = random.choice(self.towers)
               self.upgrade_tower(random_tower[1], random_tower[2])
        elif action_name == "use_ability":
            self.use_ability()

        # Check for inefficient resource usage
        if original_resources > self.resources:
            if original_resources == self.resources:
                reward += PENALTY_WASTED_RESOURCES

        self.spawn_enemies()
        self.move_enemies()
        self.enemies_attack_towers() # Gegner greifen Türme an
        self.handle_tower_attacks()

        # Überprüfe, welche Gegner entfernt werden müssen
        enemies_to_remove = [i for i, enemy in enumerate(self.enemies) if enemy[3] <= 0]

        # Berechne die Belohnung für das Töten von Gegnern
        for _ in enemies_to_remove:
           reward += REWARD_ENEMY_KILL

        # Entferne die Gegner
        for enemy_idx in reversed(enemies_to_remove):
            if enemy_idx < len(self.enemies):
                self.enemies.pop(enemy_idx)

        # Check if all enemies are gone
        if len(self.enemies) == 0 and self.wave > 1:
            reward += REWARD_WAVE_COMPLETE
            #self.wave += 1 # wave is already increased when spawning new enemies

        reward += self.base_hp * REWARD_BASE_SURVIVAL # reward remaining base health
        return self.get_state(), reward, self.done

    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Zeichne das Grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1) # Draw grid
                # Zeichne Turm falls vorhanden
                if self.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0,255,0), rect) # Turm ist grün

        # Zeichne Türme mit Lebensbalken und Reichweite
        for tower in self.towers:
            range_color = (0, 0, 255)  # Blau für die Reichweite
            tower_type = tower[0]
            x, y = tower[1], tower[2]

             # Reichweite je nach Turmtyp
            tower_range = 100 if tower_type == "basic" else \
                          150 if tower_type == "laser" else \
                          80 if tower_type == "slow" else \
                          70 if tower_type == "area" else 120

            pygame.draw.circle(self.screen, range_color, (x, y), tower_range, 1)  # Zeichne Reichweite

            pygame.draw.circle(self.screen, (0, 255, 0), (tower[1], tower[2]), 10) # Turm ist grün
            # Lebensbalken
            hp_percentage = tower[4] / 100 # Prozentuale Lebensanzeige
            bar_width = int(20 * hp_percentage)
            pygame.draw.rect(self.screen, (255, 0, 0), (tower[1] - 10, tower[2] - 15, 20, 5)) # Hintergrund
            pygame.draw.rect(self.screen, (0, 255, 0), (tower[1] - 10, tower[2] - 15, bar_width, 5))  # HP-Balken


        # Zeichne Gegner mit Lebensbalken
        for enemy in self.enemies:
           pygame.draw.circle(self.screen, (255, 0, 0), (enemy[1], enemy[2]), 8)  # Gegner zeichnen
           # Lebensbalken
           hp_percentage = enemy[3] / 100  # Gesundheit als Prozentwert
           bar_width = int(20 * hp_percentage)
           pygame.draw.rect(self.screen, (255, 0, 0), (enemy[1] - 10, enemy[2] - 15, 20, 5))  # Hintergrund
           pygame.draw.rect(self.screen, (0, 255, 0), (enemy[1] - 10, enemy[2] - 15, bar_width, 5))  # HP-Balken


        font = pygame.font.Font(None, 20)
        text_res = font.render(f"Resources: {self.resources}", True, (255, 255, 255))
        text_wave = font.render(f"Wave: {self.wave}", True, (255,255,255))
        text_hp = font.render(f"HP: {self.base_hp}", True, (255,255,255))

        self.screen.blit(text_res, (10, 10))
        self.screen.blit(text_wave, (10, 30))
        self.screen.blit(text_hp, (10, 50))

        if self.resources <= 0:
            font = pygame.font.Font(None, 30)
            warning_text = font.render("Nicht genug Ressourcen!", True, (255, 0, 0))
            self.screen.blit(warning_text, (WIDTH // 2 - 100, HEIGHT // 2 - 20))

        pygame.display.flip()
        self.clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0


    def add(self, state, action, reward, next_state, done, error):
      max_priority = max(self.priorities, default=1) if self.buffer else 1

      self.buffer.append((state, action, reward, next_state, done))
      self.priorities.append(max_priority)

    def update_beta(self):
      self.frame +=1
      self.beta = min(1.0, self.beta_start + (self.frame * (1.0 - self.beta_start) / self.beta_frames))


    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + PRIORITIZED_REPLAY_EPS) ** self.alpha


    def sample(self, batch_size):
      if len(self.buffer) < batch_size:
        return None
      
      self.update_beta()

      priorities = np.array(self.priorities)
      probs = priorities / priorities.sum()
      
      indices = np.random.choice(len(self.buffer), batch_size, p=probs)
      samples = [self.buffer[idx] for idx in indices]

      states = np.array([sample[0] for sample in samples])
      actions = np.array([sample[1] for sample in samples])
      rewards = np.array([sample[2] for sample in samples])
      next_states = np.array([sample[3] for sample in samples])
      dones = np.array([sample[4] for sample in samples])

      weights = (len(self.buffer) * probs[indices]) ** -self.beta
      weights = weights / weights.max()

      return states, actions, rewards, next_states, dones, indices, weights
    
    def __len__(self):
        return len(self.buffer)

@tf.keras.utils.register_keras_serializable()
def custom_output(tensor):
    value, advantages = tf.split(tensor, num_or_size_splits=[1, NUM_ACTIONS], axis=-1)
    return value + (advantages - tf.reduce_mean(advantages, axis=-1, keepdims=True))

def masked_loss_function(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return loss * mask

class DDQNAgent:
    def __init__(self, state_size, action_size, alpha=0.6, beta_start=0.4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(max_size=MEMORY_SIZE_MAX, alpha=alpha, beta_start=beta_start)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_rate = EPSILON_DECAY_RATE
        self.target_update_freq = TARGET_UPDATE_FREQ
        self.model = self.load_or_build_model()
        self.target_model = self.load_or_build_model()
        self.update_target_model()
        self.training_step = 0

    def load_or_build_model(self):
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
        else:
            return self.build_model()
    
    def build_model(self):
        input_layer = Input(shape=(self.state_size,))
        
        # Shared layers with Layer Normalization
        shared_dense_1 = Dense(128, activation='relu')(input_layer)
        shared_norm_1 = LayerNormalization()(shared_dense_1)
        
        shared_dense_2 = Dense(128, activation='relu')(shared_norm_1)
        shared_norm_2 = LayerNormalization()(shared_dense_2)

        # Value stream
        value_dense_1 = Dense(64, activation='relu')(shared_norm_2)
        value_norm_1 = LayerNormalization()(value_dense_1)
        value_output = Dense(1, activation='linear')(value_norm_1)

        # Advantage stream
        advantage_dense_1 = Dense(64, activation='relu')(shared_norm_2)
        advantage_norm_1 = LayerNormalization()(advantage_dense_1)
        advantage_output = Dense(self.action_size, activation='linear')(advantage_norm_1)
        
        # Combine streams
        output_layer = concatenate([value_output, advantage_output])
        output = Lambda(custom_output, output_shape=(self.action_size,))(output_layer)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model


    def remember(self, state, action, reward, next_state, done, error):
        self.memory.add(state, action, reward, next_state, done, error)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_reshaped = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state_reshaped, verbose=0)

        return np.argmax(act_values[0])

    def replay(self, batch_size):
      
      sampled_data = self.memory.sample(batch_size)

      if sampled_data is None:
         return
      
      states, actions, rewards, next_states, dones, indices, weights = sampled_data

      # Predict Q-values for current states and next states using batches
      q_values = self.model.predict(states, verbose=0)
      next_q_values = self.target_model.predict(next_states, verbose=0)

      errors = []
      targets = np.zeros_like(q_values) # Initialize targets

      for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
          if done:
              target_q = reward # If done, target is simply the reward
          else:
              target_q = reward + self.gamma * np.max(next_q_values[i]) # If not done, use next_q_value
            
          errors.append(abs(q_values[i][action] - target_q)) # Calculate error
          targets[i][action] = target_q  # Store the target

      loss = self.model.train_on_batch(states, targets, sample_weight=weights)
      self.memory.update_priorities(indices, errors)

      
      # Epsilon Decay Annealing
      self.epsilon = max(self.epsilon_min, EPSILON_START - self.training_step * self.epsilon_decay_rate)
      self.training_step +=1


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def save_model(self):
        self.model.save(MODEL_PATH)

def test_environment():
    env = TowerDefenseEnvironment()
    env.reset()
    for _ in range(200):
      env.handle_events()
      env.render()
      action = random.choice(range(NUM_ACTIONS))  # Zufällige Aktionen
      _, _, done = env.step(action)
      if done:
          break
      time.sleep(0.05)

    print("Environment Test abgeschlossen.")

def train_agent(alpha=0.6, beta_start=0.4):
    env = TowerDefenseEnvironment()
    action_size = NUM_ACTIONS
    agent = DDQNAgent(env.calculate_state_size(), action_size, alpha=alpha, beta_start=beta_start) # initial state size
    episode_rewards = []
    
    for e in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            # Calculate error for the prioritized replay buffer
            target = reward
            if not done:
              if next_state is not None:
                   # We predict the next state q-values, but not inside the loop
                  target = reward + GAMMA * np.max(agent.target_model.predict(np.reshape(next_state, [1, agent.state_size]), verbose=0)) # Use stored value
            
            # We predict the current state q-values, but not inside the loop
            current_q_value = agent.model.predict(np.reshape(state, [1, agent.state_size]), verbose=0)[0][action]
            error = abs(current_q_value - target) 

            agent.remember(state, action, reward, next_state, done, error)
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps % RENDER_EVERY == 0:
               env.handle_events()
               env.render()

        end_time = time.time()
        episode_duration = end_time - start_time

        episode_rewards.append(total_reward)
        print(
            f"Episode: {e + 1}/{NUM_EPISODES}, Belohnung: {total_reward:.2f}, Schritte: {steps}, Epsilon: {agent.epsilon:.2f}, Dauer: {episode_duration:.2f}s, Memory size: {len(agent.memory)}"
        )

        if len(agent.memory) > BATCH_SIZE: # Train nur wenn genug Daten im Memory sind
            agent.replay(BATCH_SIZE)
        
        if e % agent.target_update_freq == 0:
             agent.update_target_model()

        agent.save_model()

    print(f"Training abgeschlossen. Durchschnittliche Belohnung: {np.mean(episode_rewards):.2f}")
    return np.mean(episode_rewards)

if __name__ == "__main__":
    #episode_rewards = train_agent(alpha=0.5, beta_start=0.3)
    
    #plt.plot(episode_rewards)
    #plt.title("Reward-Verlauf")
    #plt.xlabel("Episode")
    #plt.ylabel("Reward")
    #plt.show()
    # 1. Umgebungsprüfung
    test_environment()

    # 2. Visualisierung: In der Testumgebung und während des Trainings
    # (siehe train_agent, wo RENDER_EVERY verwendet wird)
    
    # 3. Test verschiedener Buffer (Priorisiert vs. Normal) - Hier wird Priorized genutzt
    # Test mit unterschiedlichen Alpha und Beta Werten
    best_reward = float('-inf')
    best_params = None
    
    # Beispiel für die Suche von Hyperparametern:
    alpha_values = [0.5, 0.6, 0.7]
    beta_start_values = [0.3, 0.4, 0.5]

    for alpha in alpha_values:
      for beta_start in beta_start_values:
        print(f"\nTraining mit alpha: {alpha}, beta_start: {beta_start}")
        avg_reward = train_agent(alpha=alpha, beta_start=beta_start)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = (alpha, beta_start)
            print("BESTER REWARD BIS JETZT: ", best_reward)


    print("\nBestes Training:")
    print(f"Alpha: {best_params[0]}, Beta_start: {best_params[1]}, Durchschnittlicher Reward: {best_reward:.2f}")


    # 4. Epsilon-Decay-Rate wird bereits in der Konfiguration oben angepasst
    
    # 5. Testen: siehe das gesamte train_agent()
