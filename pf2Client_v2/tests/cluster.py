import random

class Particle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Particle({self.x}, {self.y}, {self.z})"

def generate_particles(n):
    particles = []
    for i in range(n):
        x = random.randint(0, 10)
        y = random.randint(0, 10)
        z = random.randint(0, 10)
        particle = Particle(x, y, z)
        particles.append(particle)
    return particles

def find_clusters(particles, radius):
    clusters = []
    while particles:
        seed = particles.pop(0)
        cluster = [seed]
        i = 0
        while i < len(cluster):
            p = cluster[i]
            neighbors = [n for n in particles if distance(p, n) < radius]
            cluster += neighbors
            for n in neighbors:
                particles.remove(n)
            i += 1
        # calculate centroid of cluster
        cx = sum(p.x for p in cluster) / len(cluster)
        cy = sum(p.y for p in cluster) / len(cluster)
        cz = sum(p.z for p in cluster) / len(cluster)
        centroid = Particle(cx, cy, cz)
        clusters.append(centroid)
    return clusters

def distance(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

particles = generate_particles(20)
print("Particles:")
print(particles)

clusters = find_clusters(particles, 3)
print("Clusters:")
print(clusters)
