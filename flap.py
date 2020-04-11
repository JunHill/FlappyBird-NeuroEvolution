import pygame
import numpy as np
import random
import math

WIN_Width = 600
WIN_Height = 600
gap = 150
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
win = pygame.display.set_mode((WIN_Width, WIN_Height))


def mutate_func(x, rate):
    for i in range(0, len(x)):
        for k in range(0, len(x[0])):
            if random.uniform(0, 1) < rate:
                x[i][k] += np.random.normal(0, 0.1, 1)[0]
    return x


def activation_func(x):
    xx = x
    for i in range(0, len(xx)):
        if xx[i] <= 0:
            xx[i] = 0
    return xx


def foo(x):
    return x*2 - 1


class NeuralNetwork:
    def __init__(self, inputs=None, hidden=None, outputs=None):
        if inputs:
            self.w1 = foo(np.random.rand(inputs, hidden))
            self.b1 = np.zeros((hidden, 1)) + 1
            self.w2 = foo(np.random.rand(hidden, outputs))
            self.b2 = np.zeros((outputs, 1)) + 1
        else:
            self.w1 = []
            self.b1 = []
            self.w2 = []
            self.b1 = []

    def mutate(self, rate):
        self.w1 = mutate_func(self.w1, rate)
        self.b1 = mutate_func(self.b1, rate)
        self.w2 = mutate_func(self.w2, rate)
        self.b2 = mutate_func(self.b2, rate)

    def copy(self):
        duplication = NeuralNetwork()
        duplication.w1 = self.w1.copy()
        duplication.b1 = self.b1.copy()
        duplication.w2 = self.w2.copy()
        duplication.b2 = self.b2.copy()
        return duplication

    @staticmethod
    def predict(self, inputs):
        inp = np.zeros((len(inputs), 1))
        for i in range(len(inputs)-1, -1, -1):
            inp[i][0] = inputs[i]

        z1 = np.dot(np.transpose(self.w1), inp) + self.b1
        a1 = activation_func(z1)
        z2 = np.dot(np.transpose(self.w2), a1) + self.b2
        a2 = activation_func(z2)

        out = []
        for i in range(len(a2)-1, -1, -1):
            out.append(a2[i])
        return out


def draw(birds, pipes):
    win.fill(BLACK)
    for i in range(0, len(birds)):
        birds[i].draw()
    for i in range(0, len(pipes)):
        pipes[i][0].draw()
        pipes[i][1].draw()
    pygame.display.flip()


def populate(birds, gen, best_bird, best_gen):
    pop = len(birds)
    birds = sorted(birds, key=lambda t: t.score, reverse=True)
    birds[pop-1] = best_bird
    print(f'best: {best_bird.score}  bird[0]: {birds[0].score}')
    if birds[0].score > best_bird.score:
        best_gen = gen
        best_bird = birds[0]

    x = 0
    fitness = []
    for i in range(0, pop):
        x += birds[i].score
        fitness.append(birds[i].score)
    new_birds = [None] * pop
    print(fitness)
    for i in range(0, pop):
        rand = rand_bird(x, fitness)
        new_brain = birds[rand].brain.copy()
        new_brain.mutate(0.1)
        new_birds[i] = Bird(new_brain)
    return new_birds, best_bird, best_gen


def step(pipes):
    for pipe in pipes:
        pipe[0].x -= 5
        pipe[1].x -= 5
        if pipe[0].x + pipe[0].w <= 0:
            pipes.remove(pipe)


def rand_bird(x, fitness):
    if x:
        ran = random.randrange(0, x)
        i = 0
        while i < len(fitness) and ran >= 0:
            ran -= fitness[i]
            i += 1
        return i - 1
    else:
        return 0


def create_pipe(pipes):
    h = math.floor(random.randrange(50, WIN_Height - gap - 50))
    pipes.append((Pipe(True, h), Pipe(False, WIN_Height - h - gap)))


class Pipe:
    def __init__(self, upper, h):
        self.upper = upper
        self.x = WIN_Width
        self.y = 0 if upper else WIN_Height - h
        self.w = 50
        self.h = h
        self.body = pygame.Surface((self.w, self.h))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def draw(self):
        self.body.fill(WHITE)
        win.blit(self.body, (self.x, int(self.y)))

    def collide(self, bird):
        return self.get_rect().colliderect(bird.get_rect())


class Bird:
    def __init__(self, brain=None):
        self.x = 100
        self.y = WIN_Height/2
        self.r = 15
        self.v = 0
        self.crashed = False
        self.score = 0
        self.body = pygame.Surface((self.r, self.r))
        self.brain = brain if brain else NeuralNetwork(4, 6, 2)
        self.color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))

    def get_rect(self):
        return pygame.Rect(self.x, int(self.y), self.r, self.r)

    def draw(self):
        self.body.fill(self.color)
        win.blit(self.body, (self.x, int(self.y)))

    def think(self, pipes):
        nearest_pipe = None
        for pipe in pipes:
            if pipe[0].x + pipe[0].w > self.x:
                nearest_pipe = pipe
                break
        x = nearest_pipe[0].x/WIN_Width
        y = nearest_pipe[0].h/WIN_Height
        return self.brain.predict(self.brain, [self.y/WIN_Height, self.v/20, x, y])

    def check_dead(self, pipes):
        if self.y + self.r > WIN_Height:
            self.crashed = True
            self.y = WIN_Height - self.r
            return True
        if self.y - self.r < 0:
            self.crashed = True
            self.y = 0
            return True
        if pipes[0][0].collide(self) or pipes[0][1].collide(self):
            self.crashed = True
            self.x -= 5
            return True
        return False

    def step(self, pipes):
        if self.crashed:
            self.x -= 5
        else:
            self.v += 1
            thinking = self.think(pipes)

            if thinking[0] > thinking[1]:
                self.v -= 12
            self.y += self.v
            if self.check_dead(pipes):
                return
            self.score += 1


def main():
    pygame.init()
    fps = pygame.time.Clock()
    size = 500
    birds = [None] * size
    pipes = []
    crashed_birds = 0
    gen = 0
    best_gen = 0
    best_bird = Bird()
    pipe_spawn = 0
    already_dead = [False] * size
    for i in range(0, size):
        birds[i] = Bird()
    create_pipe(pipes)
    draw(birds, pipes)

    run = True
    while run:
        fps.tick(70)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if crashed_birds >= len(birds):
            win.fill(BLACK)
            pygame.display.flip()
            already_dead = [0] * size
            temp = populate(birds, gen, best_bird, best_gen)
            birds = temp[0]
            best_bird = temp[1]
            best_gen = temp[2]
            pipes = []
            create_pipe(pipes)
            pipe_spawn = 0
            gen += 1
            crashed_birds = 0
            print(f"gen: {gen}, bestgen: {best_gen}({best_bird.score})")
        else:
            step(pipes)
            for i in range(0, size):
                birds[i].step(pipes)
                if birds[i].crashed and not already_dead[i]:
                    crashed_birds += 1
                    already_dead[i] = True
            pipe_spawn += 1
            if pipe_spawn == 50:
                pipe_spawn = 0
                create_pipe(pipes)
        draw(birds, pipes)
    pygame.quit()


main()





