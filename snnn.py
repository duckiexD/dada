import numpy as np
import pygame
import random
import math
from collections import deque
import time

pygame.init()

# Константы
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 10


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

class LearningSnake:
    def __init__(self, brain=None, generation=1):
        self.body = deque([(GRID_WIDTH // 2, GRID_HEIGHT // 2)])
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.food = self.spawn_food()
        self.score = 0
        self.steps = 0
        self.alive = True
        self.fitness = 0
        self.generation = generation
        
        if brain is None:
            self.brain = {
                'food_preference': 0.0, 
                'safety_preference': 0.0,  
                'exploration': 0.0,  
            }
        else:
            self.brain = brain.copy()
    
    def spawn_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.body:
                return food
    
    def get_vision(self):
        head_x, head_y = self.body[0]
        
        directions = [
            self.direction,
            (-self.direction[1], self.direction[0]),
            (self.direction[1], -self.direction[0])
        ]
        
        vision = []
        
        for dx, dy in directions:
            # Проверка стен
            new_x = head_x + dx
            new_y = head_y + dy
            wall_detected = 1 if (new_x < 0 or new_x >= GRID_WIDTH or new_y < 0 or new_y >= GRID_HEIGHT) else 0
            
            # Проверка тела
            body_detected = 1 if ((new_x, new_y) in list(self.body)[1:]) else 0
            
            # Определение направления к еде
            food_dx = self.food[0] - head_x
            food_dy = self.food[1] - head_y
            
            # Нормализация направления к еде
            food_dir_x = 1 if food_dx > 0 else -1 if food_dx < 0 else 0
            food_dir_y = 1 if food_dy > 0 else -1 if food_dy < 0 else 0
            
            # Сравнение с текущим направлением
            moving_toward_food = 1 if (dx == food_dir_x and dy == food_dir_y) else 0
            
            vision.extend([wall_detected, body_detected, moving_toward_food])
        
        return vision
    
    def decide_direction(self):
        if self.brain['food_preference'] == 0.0 and self.brain['safety_preference'] == 0.0 and self.brain['exploration'] == 0.0:
            return random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        
        vision = self.get_vision()
        scores = [0, 0, 0] 
        
        for i in range(3):
            wall_idx = i * 3
            body_idx = i * 3 + 1
            food_idx = i * 3 + 2
            
            # Используем параметры мозга для принятия решений
            wall_penalty = -vision[wall_idx] * self.brain['safety_preference'] * 10
            body_penalty = -vision[body_idx] * self.brain['safety_preference'] * 8
            food_score = vision[food_idx] * self.brain['food_preference'] * 6
            
            # Исследование добавляет случайность
            exploration_bonus = random.uniform(-1, 1) * self.brain['exploration'] * 2
            
            scores[i] = wall_penalty + body_penalty + food_score + exploration_bonus
        
        # Выбираем направление с лучшей оценкой
        best_direction_idx = scores.index(max(scores))
        
        if best_direction_idx == 0:
            return self.direction  # Вперед
        elif best_direction_idx == 1:
            return (-self.direction[1], self.direction[0])  # Влево
        else:
            return (self.direction[1], -self.direction[0])  # Вправо
    
    def move(self):
        if not self.alive:
            return
        
        self.steps += 1
        
        # Решаем куда двигаться
        self.direction = self.decide_direction()
        
        head_x, head_y = self.body[0]
        new_x = head_x + self.direction[0]
        new_y = head_y + self.direction[1]
        new_head = (new_x, new_y)
        
        # Проверяем столкновение со стеной
        if new_x < 0 or new_x >= GRID_WIDTH or new_y < 0 or new_y >= GRID_HEIGHT:
            self.alive = False
            return
        
        # Проверяем столкновение с собой
        if new_head in list(self.body)[1:]:
            self.alive = False
            return
        
        self.body.appendleft(new_head)
        
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.body.pop()
    
    def get_fitness(self):
        return self.score * 100 + self.steps

class GeneticAlgorithm:
    def __init__(self, population_size=15):
        self.population_size = population_size
        self.population = [LearningSnake(generation=1) for _ in range(population_size)]
        self.generation = 1
        self.best_snakes = []
        self.best_scores = []
    
    def evaluate_population(self):
        max_steps_per_snake = 100
        best_generation_score = 0
        best_generation_snake = None
        
        for snake in self.population:
            steps = 0
            while snake.alive and steps < max_steps_per_snake:
                snake.move()
                steps += 1
            
            if snake.score > best_generation_score:
                best_generation_score = snake.score
                best_generation_snake = snake
        
        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
        
        # Сохраняем лучшую змейку поколения
        if best_generation_snake:
            best_snake_copy = LearningSnake(best_generation_snake.brain, self.generation)
            best_snake_copy.score = best_generation_score
            self.best_snakes.append(best_snake_copy)
            self.best_scores.append(best_generation_score)
    
    def improve_brain(self, brain, current_generation):
        """Улучшаем мозг в зависимости от номера поколения"""
        improved_brain = brain.copy()
        
        target_knowledge = min(1.0, (current_generation - 1) * 0.15)
        
        # Плавно приближаемся к целевому уровню знаний
        for key in improved_brain:
            current = brain[key]
            if target_knowledge > current:
                improved_brain[key] = min(1.0, current + (target_knowledge - current) * 0.5)
            
            # Добавляем небольшие случайные вариации (только для поколений > 1)
            if current_generation > 1 and random.random() < 0.2:
                improved_brain[key] = max(0.0, min(1.0, improved_brain[key] + random.uniform(-0.05, 0.1)))
        
        return improved_brain
    
    def create_new_generation(self):
        elite_size = max(2, self.population_size // 3)
        elites = self.population[:elite_size]
        
        new_population = []
        
        # Улучшаем и сохраняем элиту для СЛЕДУЮЩЕГО поколения
        for elite in elites:
            improved_brain = self.improve_brain(elite.brain, self.generation + 1)
            new_snake = LearningSnake(improved_brain, self.generation + 1)
            new_population.append(new_snake)
        
        # Создаем разнообразное потомство для СЛЕДУЮЩЕГО поколения
        while len(new_population) < self.population_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            # Скрещивание - усреднение с улучшением
            child_brain = {}
            for key in parent1.brain.keys():
                base_value = (parent1.brain[key] + parent2.brain[key]) / 2
                # Улучшаем потомство для следующего поколения
                target_knowledge = min(1.0, self.generation * 0.15)
                if target_knowledge > base_value:
                    child_brain[key] = min(1.0, base_value + (target_knowledge - base_value) * 0.3)
                else:
                    child_brain[key] = base_value
            
            new_snake = LearningSnake(child_brain, self.generation + 1)
            new_population.append(new_snake)
        
        self.population = new_population
        self.generation += 1
    
    def run_generation(self):
        self.evaluate_population()
        self.create_new_generation()

def run_learning_demo(ga, max_generations=10):
    """Демонстрация обучения змейки от 1 до 10 поколения"""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    current_generation = 0
    running = True
    
    while running and current_generation < max_generations:
        if current_generation < len(ga.best_snakes):
            original_snake = ga.best_snakes[current_generation]
            generation_best_score = ga.best_scores[current_generation]
        else:
            # Если что-то пошло не так, создаем змейку с нулевыми знаниями
            brain = {
                'food_preference': 0.0,
                'safety_preference': 0.0,
                'exploration': 0.0,
            }
            original_snake = LearningSnake(brain, current_generation + 1)
            generation_best_score = 0
        
        start_time = time.time()
        demo_duration = 8
        current_demo_score = 0
        
        # Создаем рабочую копию змейки для демонстрации
        demo_snake = LearningSnake(original_snake.brain, original_snake.generation)
        
        while running and time.time() - start_time < demo_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        break
            
            if demo_snake.alive:
                demo_snake.move()
                current_demo_score = demo_snake.score
            else:
                # Перезапускаем с тем же мозгом (теми же знаниями)
                demo_snake = LearningSnake(original_snake.brain, original_snake.generation)
                current_demo_score = 0
            
            screen.fill(BLACK)
            
            # Сетка
            for x in range(0, WIDTH, GRID_SIZE):
                pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT), 1)
            for y in range(0, HEIGHT, GRID_SIZE):
                pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y), 1)
            
            # Стены
            pygame.draw.rect(screen, BLUE, (0, 0, WIDTH, HEIGHT), 3)
            
            # Еда
            food_rect = pygame.Rect(
                demo_snake.food[0] * GRID_SIZE,
                demo_snake.food[1] * GRID_SIZE,
                GRID_SIZE, GRID_SIZE
            )
            pygame.draw.rect(screen, RED, food_rect)
            
            # Змейка
            for i, segment in enumerate(demo_snake.body):
                color_intensity = 255 - (i * 50 // len(demo_snake.body))
                color = (0, min(255, color_intensity), 0)
                
                segment_rect = pygame.Rect(
                    segment[0] * GRID_SIZE,
                    segment[1] * GRID_SIZE,
                    GRID_SIZE, GRID_SIZE
                )
                pygame.draw.rect(screen, color, segment_rect)
                pygame.draw.rect(screen, BLACK, segment_rect, 1)
            
            # Информация о поколении
            font = pygame.font.Font(None, 36)
            gen_text = font.render(f"Поколение: {current_generation + 1}/10", True, YELLOW)
            screen.blit(gen_text, (10, 10))
            
            score_text = font.render(f"Текущий счет: {current_demo_score}", True, WHITE)
            screen.blit(score_text, (10, 50))
            
            best_text = font.render(f"Лучший счет: {generation_best_score}", True, WHITE)
            screen.blit(best_text, (10, 90))
            
            # Прогресс обучения
            brain_font = pygame.font.Font(None, 28)
            
            # Стремление к еде
            food_progress = original_snake.brain['food_preference']
            food_text = brain_font.render(f"Знание о еде: {food_progress:.1%}", True, ORANGE)
            screen.blit(food_text, (10, 140))
            
            # Прогресс-бар стремления к еде
            food_bar_rect = pygame.Rect(200, 145, 200, 20)
            pygame.draw.rect(screen, GRAY, food_bar_rect, 1)
            food_fill_rect = pygame.Rect(202, 147, int(196 * food_progress), 16)
            pygame.draw.rect(screen, ORANGE, food_fill_rect)
            
            # Осторожность
            safety_progress = original_snake.brain['safety_preference']
            safety_text = brain_font.render(f"Осторожность: {safety_progress:.1%}", True, BLUE)
            screen.blit(safety_text, (10, 180))
            
            # Прогресс-бар осторожности
            safety_bar_rect = pygame.Rect(200, 185, 200, 20)
            pygame.draw.rect(screen, GRAY, safety_bar_rect, 1)
            safety_fill_rect = pygame.Rect(202, 187, int(196 * safety_progress), 16)
            pygame.draw.rect(screen, BLUE, safety_fill_rect)
            
            # Исследование
            explore_progress = original_snake.brain['exploration']
            explore_text = brain_font.render(f"Исследование: {explore_progress:.1%}", True, GREEN)
            screen.blit(explore_text, (10, 220))
            
            # Прогресс-бар исследования
            explore_bar_rect = pygame.Rect(200, 225, 200, 20)
            pygame.draw.rect(screen, GRAY, explore_bar_rect, 1)
            explore_fill_rect = pygame.Rect(202, 227, int(196 * explore_progress), 16)
            pygame.draw.rect(screen, GREEN, explore_fill_rect)
            
            # Общий уровень интеллекта
            total_knowledge = (food_progress + safety_progress + explore_progress) / 3
            intel_text = font.render(f"Общий интеллект: {total_knowledge:.1%}", True, YELLOW)
            screen.blit(intel_text, (10, 260))
            
            # Статус
            status_text = font.render(f"Статус: {'ЖИВА' if demo_snake.alive else 'МЕРТВА'}", True, 
                                    GREEN if demo_snake.alive else RED)
            screen.blit(status_text, (10, 300))
            
            # Особое сообщение для первого поколения
            if current_generation == 0:
                dumb_text = font.render("ПОЛНЫЙ НОЛЬ ЗНАНИЙ!", True, RED)
                screen.blit(dumb_text, (WIDTH // 2 - 150, HEIGHT // 2 - 50))
            
            # Подсказки
            hint_font = pygame.font.Font(None, 24)
            hint1 = hint_font.render("SPACE - пропустить", True, GRAY)
            hint2 = hint_font.render("ESC - выйти", True, GRAY)
            screen.blit(hint1, (WIDTH - 200, HEIGHT - 50))
            screen.blit(hint2, (WIDTH - 200, HEIGHT - 25))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        current_generation += 1
        total_knowledge = sum(original_snake.brain.values()) / 3
        print(f"Поколение {current_generation}: знания = {total_knowledge:.1%}")
    
    return current_generation

def main():
    print("🚀 Запуск эволюции змеек...")
    print("🧠 Поколение 1: Самая тупая без знаний! (0%, 0%, 0%)")
    print("📈 С каждым поколением: + к еде, + к осторожности, + к исследованию")
    print("=" * 65)
    
    ga = GeneticAlgorithm(population_size=12)
    
    # Быстрая эволюция 10 поколений
    for generation in range(10):
        ga.run_generation()
        best_score = ga.best_scores[-1] if ga.best_scores else 0
        if generation < len(ga.best_snakes):
            knowledge = sum(ga.best_snakes[generation].brain.values()) / 3
        else:
            knowledge = 0.0
        print(f"Поколение {generation + 1}: счёт = {best_score:2d} | знания = {knowledge:.1%}")
    
    print("\n🎯 Запуск демонстрации обучения...")
    print("=" * 65)
    
    run_learning_demo(ga, max_generations=10)
    
    # Финальная статистика
    print("\n" + "=" * 65)
    print("📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
    print("=" * 65)
    
    for i, (snake, score) in enumerate(zip(ga.best_snakes[:10], ga.best_scores[:10])):
        knowledge = sum(snake.brain.values()) / 3
        print(f"Поколение {i+1}: счёт = {score:2d} | знания = {knowledge:.1%}")
    
    best_overall_score = max(ga.best_scores) if ga.best_scores else 0
    best_gen = ga.best_scores.index(best_overall_score) + 1 if ga.best_scores else 1
    
    print(f"\n🏆 Лучший результат: {best_overall_score} очков (поколение {best_gen})")
    
    final_snake = ga.best_snakes[best_gen - 1]
    final_knowledge = sum(final_snake.brain.values()) / 3
    print(f"💡 Финальный уровень знаний: {final_knowledge:.1%}")
    print("=" * 65)
    
    pygame.quit()

if __name__ == "__main__":

    main()
