import numpy as np

# class explains n chain rope behaviour
TO_RAD = np.pi / 180.0
TO_DEG = 180.0 / np.pi
GRAV_CONST = 9.81

class Node:
    # Class describes rope part (one node): it contains stretching spring, and weight ball (material point)
    # which hanging on this spring
    L0 = 5.0  # rope idle length (1 meter)
    k = 10000.0  # spring (rope) stretch coefficient: F = -K*dx
    k_damp = 5.0  # damper coefficient defines damper force: F = -K*vel
    Cx = 0.05  # Aerodynamic drag coefficient
    mass = 0.1

    def __init__(self, chain_n, init_pos, ch_prev=None, ch_next=None):
        self.pos = init_pos
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])
        self.prev = ch_prev
        self.next = ch_next
        self.Fs = np.array([0.0, 0.0, 0.0])
        self.Fv = np.array([0.0, 0.0, 0.0])
        self.L = self.L0
        self.time = 0.0
        self.data = {'x': [], 'y': [], 'z': [], 'vel': [], 'time': [], 'energy': []}
        self.chain_n = chain_n
        self.energy = GRAV_CONST * self.mass * self.pos[1]
        self.step_cnt = 0

    def cycle(self, dt, record=True, record_step=1, V0=0.0):
        if record:
            self.record_cycle(record_step=record_step)

        if self.prev is None:
            prev_pos = np.array([0.0, 0.0, 0.0])
        else:
            prev_pos = self.prev.pos

        if self.next is not None:
            self.next.cycle(dt, record=record, V0=V0)
            next_Fs = self.next.Fs
            next_Fv = self.next.Fv

        else:
            next_Fs = np.array([0.0, 0.0, 0.0])
            next_Fv = np.array([0.0, 0.0, 0.0])

        # spring
        len_rope = self.pos - prev_pos
        dir_rope = len_rope / np.max([np.linalg.norm(len_rope), 0.01 * self.L0])

        dL_rope = max([np.linalg.norm(len_rope) - self.L0, 0.0])

        dL_vel = (np.linalg.norm(len_rope) - self.L) / dt

        self.L = np.linalg.norm(len_rope)

        # Forces
        self.Fs = -dL_rope * self.k * dir_rope # сила упругости
        self.Fv = np.clip(-self.k_damp * dL_vel * dir_rope * int(dL_rope > 0),
                          -self.mass * GRAV_CONST * 20.0 * 0.001 / dt, self.mass * 20.0 * GRAV_CONST * 0.001 / dt) # демпфирование

        vel_aero = self.vel + np.array([V0, 0.0, 0.0])
        F_cx = -vel_aero * self.Cx * 1.225 * 0.5 * np.linalg.norm(vel_aero)

        G = -self.mass * GRAV_CONST * np.array([0.0, 1.0, 0.0])
        F = self.Fs + self.Fv + F_cx + G - next_Fs - next_Fv

        # movement
        acc = F / self.mass

        # считаем по трапециям
        # vel = self.vel + 0.5*(self.acc + acc) * dt
        # self.pos += 0.5*(self.vel + vel) * dt

        # self.vel = vel
        self.acc = acc

        self.vel += self.acc*dt
        self.pos += self.vel*dt

        self.energy = 0.5 * self.mass * np.linalg.norm(self.vel) ** 2 + GRAV_CONST * self.mass * self.pos[1] + 0.5 * self.k * dL_rope ** 2
        self.time += dt
        self.step_cnt += 1

    def record_cycle(self, record_step=1):
        if self.step_cnt % record_step == 0:
            self.data['x'].append(self.pos[0])
            self.data['y'].append(self.pos[1])
            self.data['z'].append(self.pos[2])
            self.data['vel'].append(np.linalg.norm(self.vel))
            self.data['energy'].append(self.energy)
            self.data['time'].append(self.time)


class Rope:
    def __init__(self, init_angle=20.0, pos_point=None, n_chains=3, G_cargo=1.0, **node_kwargs):
        self.nodes = []

        self.head = None  # top node
        self.tail = None  # last node
        self.energy_0 = 0.0

        Node.L0 = node_kwargs.get('L0', 1.0)
        Node.k = node_kwargs.get('k', 10000.0)
        Node.k_damp = node_kwargs.get('k_damp', 5.0)
        Node.Cx = node_kwargs.get('Cx', 0.05)
        Node.mass = node_kwargs.get('mass', 0.1)

        for i in range(n_chains):
            if pos_point is None:
                pos = np.array(
                    [np.sin(init_angle * TO_RAD) * Node.L0 * (i + 1),
                     -np.cos(init_angle * TO_RAD) * Node.L0 * (i + 1),
                     0.0])
            else:
                pos = pos_point * (i + 1)

            if self.head is None:
                node = Node(i, pos, ch_prev=None, ch_next=None)
                self.head = node
                self.tail = node
            else:
                node = Node(i, pos, ch_prev=self.tail, ch_next=None)
                self.tail = node
                node.prev.next = node

            if i == n_chains - 1:
                node.mass = G_cargo

            self.energy_0 += GRAV_CONST * node.mass * node.L0 * (i + 1)
            self.nodes.append(node)

    @staticmethod
    def from_initial_angle(angle):
        pass

    @staticmethod
    def from_initial_pos(pos):
        pass

    def cycle(self, dt, vel0=0.0, record_step=1):
        self.head.cycle(dt, V0=vel0)

    def calculate(self, time, dt=0.001, V0=0.0, record_step=1):
        T = 0
        while T < time:
            self.cycle(dt, vel0=V0, record_step=record_step)
            T += dt

    def get_data(self):
        """
        Returns all nodes data (coordinates, absolute velocity, total energy), where each node describes by node index
        :return: data -> dict
        """
        # data = pd.DataFrame()
        data = {'time': self.nodes[0].data['time'],
                'energy': self.energy_0}

        for ix, _ in enumerate(self.nodes):
            data[f'x_{ix+1}'] = self.nodes[ix].data['x']
            data[f'y_{ix+1}'] = self.nodes[ix].data['y']
            data[f'z_{ix+1}'] = self.nodes[ix].data['z']
            data[f'vel_{ix+1}'] = self.nodes[ix].data['vel']
            data['energy'] = np.add(data['energy'], self.nodes[ix].data['energy'])

        return data
