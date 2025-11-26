import pytest
import pygame
import random
import sys
import os
from unittest.mock import Mock, patch

# Добавляем путь к исходному файлу
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем классы из основного файла
from snnn import LearningSnake, GeneticAlgorithm, GRID_WIDTH, GRID_HEIGHT

class TestLearningSnake:
    """Тесты для класса LearningSnake"""
    
    def test_initialization(self):
        """Тест инициализации змейки"""
        snake = LearningSnake()
        
        assert len(snake.body) == 1
        assert snake.body[0] == (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        assert snake.score == 0
        assert snake.steps == 0
        assert snake.alive == True
        assert 'food_preference' in snake.brain
        assert 'safety_preference' in snake.brain
        assert 'exploration' in snake.brain
    
    def test_spawn_food(self):
        """Тест генерации еды"""
        snake = LearningSnake()
        food = snake.spawn_food()
        
        assert isinstance(food, tuple)
        assert len(food) == 2
        assert 0 <= food[0] < GRID_WIDTH
        assert 0 <= food[1] < GRID_HEIGHT
        assert food not in snake.body
    
    def test_get_vision(self):
        """Тест получения данных зрения змейки"""
        snake = LearningSnake()
        snake.body = deque([(5, 5)])
        snake.direction = (1, 0)  # Движение вправо
        snake.food = (7, 5)  # Еда справа
        
        vision = snake.get_vision()
        
        assert isinstance(vision, list)
        assert len(vision) == 9  # 3 направления × 3 параметра
        
        # Проверяем структуру данных зрения
        for i in range(3):
            assert vision[i*3] in [0, 1]    # Стена
            assert vision[i*3+1] in [0, 1]  # Тело
            assert vision[i*3+2] in [0, 1]  # Направление к еде
    
    def test_move_without_collision(self):
        """Тест движения без столкновений"""
        snake = LearningSnake()
        initial_length = len(snake.body)
        initial_position = snake.body[0]
        
        # Мокаем решение направления для предсказуемого теста
        with patch.object(snake, 'decide_direction', return_value=(1, 0)):
            snake.move()
        
        assert len(snake.body) == initial_length
        assert snake.body[0] == (initial_position[0] + 1, initial_position[1])
        assert snake.alive == True
    
    def test_move_with_food_collision(self):
        """Тест движения со сбором еды"""
        snake = LearningSnake()
        snake.body = deque([(5, 5)])
        snake.food = (6, 5)  # Еда прямо перед змейкой
        
        with patch.object(snake, 'decide_direction', return_value=(1, 0)):
            snake.move()
        
        assert len(snake.body) == 2  # Должна вырасти
        assert snake.score == 1
        assert snake.food != (6, 5)  # Новая еда должна появиться
    
    def test_move_with_wall_collision(self):
        """Тест столкновения со стеной"""
        snake = LearningSnake()
        snake.body = deque([(GRID_WIDTH - 1, 5)])  # У правой стены
        snake.direction = (1, 0)  # Движение в стену
        
        snake.move()
        
        assert snake.alive == False
    
    def test_move_with_self_collision(self):
        """Тест столкновения с собой"""
        snake = LearningSnake()
        snake.body = deque([(5, 5), (4, 5), (4, 6)])  # Змейка с телом
        snake.direction = (0, -1)  # Движение в свое тело
        
        snake.move()
        
        assert snake.alive == False
    
    def test_decide_direction_random(self):
        """Тест случайного выбора направления при нулевых знаниях"""
        snake = LearningSnake()
        snake.brain = {'food_preference': 0.0, 'safety_preference': 0.0, 'exploration': 0.0}
        
        directions = set()
        for _ in range(10):
            direction = snake.decide_direction()
            directions.add(direction)
        
        # Должны быть разные направления (случайный выбор)
        assert len(directions) > 1
    
    def test_get_fitness(self):
        """Тест расчета fitness"""
        snake = LearningSnake()
        snake.score = 3
        snake.steps = 50
        
        fitness = snake.get_fitness()
        
        assert fitness == 3 * 100 + 50
        assert fitness > 0

class TestGeneticAlgorithm:
    """Тесты для класса GeneticAlgorithm"""
    
    def test_initialization(self):
        """Тест инициализации генетического алгоритма"""
        ga = GeneticAlgorithm(population_size=10)
        
        assert len(ga.population) == 10
        assert ga.generation == 1
        assert isinstance(ga.population[0], LearningSnake)
        assert ga.best_snakes == []
        assert ga.best_scores == []
    
    def test_evaluate_population(self):
        """Тест оценки популяции"""
        ga = GeneticAlgorithm(population_size=5)
        
        # Мокаем движение змеек для контролируемого тестирования
        for snake in ga.population:
            snake.move = Mock()
            snake.get_fitness = Mock(return_value=random.randint(100, 500))
        
        ga.evaluate_population()
        
        assert len(ga.best_snakes) > 0
        assert len(ga.best_scores) > 0
        # Популяция должна быть отсортирована по fitness
        assert ga.population[0].get_fitness() >= ga.population[-1].get_fitness()
    
    def test_improve_brain(self):
        """Тест улучшения мозга"""
        ga = GeneticAlgorithm()
        original_brain = {'food_preference': 0.1, 'safety_preference': 0.2, 'exploration': 0.3}
        
        improved_brain = ga.improve_brain(original_brain, current_generation=2)
        
        # Проверяем, что все значения в допустимом диапазоне
        for key, value in improved_brain.items():
            assert 0.0 <= value <= 1.0
            # Значения должны улучшиться или остаться прежними
            assert value >= original_brain[key] - 0.05  # Учитываем возможные случайные изменения
    
    def test_create_new_generation(self):
        """Тест создания нового поколения"""
        ga = GeneticAlgorithm(population_size=6)
        ga.generation = 1
        
        # Создаем mock популяцию с разными fitness
        for i, snake in enumerate(ga.population):
            snake.get_fitness = Mock(return_value=100 - i * 10)
            snake.brain = {
                'food_preference': 0.1 + i * 0.1,
                'safety_preference': 0.2 + i * 0.1,
                'exploration': 0.3 + i * 0.1
            }
        
        ga.evaluate_population()
        original_generation = ga.generation
        original_population_size = len(ga.population)
        
        ga.create_new_generation()
        
        assert ga.generation == original_generation + 1
        assert len(ga.population) == original_population_size
        assert all(snake.generation == ga.generation for snake in ga.population)
    
    def test_run_generation(self):
        """Тест выполнения одного поколения"""
        ga = GeneticAlgorithm(population_size=3)
        initial_generation = ga.generation
        
        # Мокаем методы для изоляции теста
        ga.evaluate_population = Mock()
        ga.create_new_generation = Mock()
        
        ga.run_generation()
        
        ga.evaluate_population.assert_called_once()
        ga.create_new_generation.assert_called_once()

class TestIntegration:
    """Интеграционные тесты"""
    
    def test_snake_lifecycle(self):
        """Тест полного жизненного цикла змейки"""
        snake = LearningSnake()
        
        # Имитируем несколько ходов
        for _ in range(5):
            if snake.alive:
                snake.move()
        
        # Проверяем, что змейка либо жива, либо умерла по правильным причинам
        if not snake.alive:
            head = snake.body[0]
            # Проверяем, что смерть произошла из-за столкновения
            assert (head[0] < 0 or head[0] >= GRID_WIDTH or 
                   head[1] < 0 or head[1] >= GRID_HEIGHT or
                   head in list(snake.body)[1:])
    
    def test_genetic_algorithm_multiple_generations(self):
        """Тест нескольких поколений эволюции"""
        ga = GeneticAlgorithm(population_size=4)
        
        # Запускаем 3 поколения
        for generation in range(3):
            ga.run_generation()
            
            assert ga.generation == generation + 2  # +1 потому что начали с 1
            assert len(ga.population) == 4
            assert len(ga.best_snakes) == generation + 1
            assert len(ga.best_scores) == generation + 1
        
        # Проверяем, что знания улучшаются со временем
        early_knowledge = sum(ga.best_snakes[0].brain.values()) / 3
        late_knowledge = sum(ga.best_snakes[-1].brain.values()) / 3
        
        # Знания должны улучшаться (или оставаться прежними)
        assert late_knowledge >= early_knowledge - 0.1  # Учитываем случайные колебания

class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_snake_spawn_food_no_space(self):
        """Тест генерации еды когда нет места (теоретически)"""
        snake = LearningSnake()
        
        # Заполняем все поле телом змейки (в теории)
        # На практике это невозможно из-за размера поля
        with pytest.raises(Exception):
            while True:
                snake.body.append((0, 0))  # Попытка создать бесконечный цикл
                # Этот тест должен показать, что нужна защита от бесконечного цикла
    
    def test_snake_initial_knowledge_zero(self):
        """Тест что начальные знания действительно нулевые"""
        snake = LearningSnake()
        
        for knowledge in snake.brain.values():
            assert knowledge == 0.0
    
    def test_genetic_algorithm_small_population(self):
        """Тест работы с очень маленькой популяцией"""
        ga = GeneticAlgorithm(population_size=1)
        
        # Должен работать без ошибок
        ga.run_generation()
        
        assert ga.generation == 2
        assert len(ga.population) == 1
    
    def test_snake_vision_at_edges(self):
        """Тест зрения змейки у краев поля"""
        snake = LearningSnake()
        snake.body = deque([(0, 0)])  # В левом верхнем углу
        snake.direction = (0, -1)  # Движение вверх (в стену)
        
        vision = snake.get_vision()
        
        # Должны обнаружить стену в некоторых направлениях
        assert 1 in vision  # Должна быть хотя бы одна стена

# Фикстуры для pytest
@pytest.fixture
def basic_snake():
    """Базовая змейка для тестов"""
    return LearningSnake()

@pytest.fixture
def genetic_algorithm():
    """Генетический алгоритм для тестов"""
    return GeneticAlgorithm(population_size=5)

# Запуск тестов
if __name__ == "__main__":
    # Запускаем тесты с verbose выводом
    pytest.main(["-v", __file__])
