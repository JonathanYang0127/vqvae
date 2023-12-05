from dataclasses import dataclass, field
import os
import sys
import h5py
import numpy as np
import torch
import json

@dataclass
class DatasetConfig:
    path: str
    rl_camera: str = "robot0_eye_in_hand"
    num_data: int = -1
    max_len: int = -1
    eval_episode_len: int = 300

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")


class RobomimicDataset:
    def __init__(self, cfg: DatasetConfig):
        config_path = os.path.join(os.path.dirname(cfg.path), "env_cfg.json")
        self.env_config = json.load(open(config_path, "r"))
        self.task_name = self.env_config["env_name"]
        self.robot = self.env_config["env_kwargs"]["robots"]
        self.cfg = cfg

        self.data = []
        datafile = h5py.File(cfg.path)
        num_episode: int = len(list(datafile["data"].keys()))  # type: ignore
        print(f"Raw Dataset size (#episode): {num_episode}")

        self.ctrl_delta = self.env_config["env_kwargs"]["controller_configs"]["control_delta"]

        self.idx2entry = []  # store idx -> (episode_idx, timestep_idx)
        episode_lens = []
        all_actions = []  # for logging purpose
        for episode_id in range(num_episode):
            if cfg.num_data > 0 and len(episode_lens) >= cfg.num_data:
                break

            episode_tag = f"demo_{episode_id}"
            episode = datafile[f"data/{episode_tag}"]
            actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore
            actions = torch.from_numpy(actions)
            all_actions.append(actions)
            episode_data: dict = {"action": actions}

            for camera in self.cfg.rl_cameras:
                obses: np.ndarray = episode[f"obs/{camera}_image"]  # type: ignore
                assert obses.shape[0] == actions.shape[0]
                episode_data[camera] = obses

            episode_len = actions.shape[0]
            if self.cfg.max_len > 0 and episode_len > self.cfg.max_len:
                print(f"removing {episode_tag} because it is too long {episode_len}")
                continue
            episode_lens.append(episode_len)

            # convert the data to list of dict
            episode_entries = []
            for i in range(episode_len):
                entry = {"action": episode_data["action"][i]}
                if self.env_config["env_kwargs"]["controller_configs"]["control_delta"]:
                    assert entry["action"].min() >= -1
                    assert entry["action"].max() <= 1

                for camera in cfg.rl_cameras:
                    entry[camera] = torch.from_numpy(episode_data[camera][i])

                self.idx2entry.append((len(self.data), len(episode_entries)))
                episode_entries.append(entry)
            self.data.append(episode_entries)
        datafile.close()

        self.obs_shape = self.data[-1][-1][cfg.rl_cameras[0]].size()
        self.action_dim = self.data[-1][-1]["action"].size()[0]
        print(f"Dataset size: {len(self.data)} episodes, {len(self.idx2entry)} steps")
        print(f"average length {np.mean(episode_lens):.1f}")
        print(f"obs shape:", self.obs_shape)
        all_actions = torch.cat(all_actions, dim=0)
        action_mins = all_actions.min(dim=0)[0]
        action_maxs = all_actions.max(dim=0)[0]
        for i in range(self.action_dim):
            print(f"action dim {i}: [{action_mins[i].item():.2f}, {action_maxs[i].item():.2f}]")

        self.env_params = dict(
            env_name=self.task_name,
            robots=self.robot,
            episode_length=cfg.eval_episode_len,
            reward_shaping=False,
            image_size=224,
            rl_image_size=self.obs_shape[-1],
            camera_names=cfg.rl_cameras,
            rl_cameras=cfg.rl_cameras,
            device="cuda",
            ctrl_delta=bool(self.env_config["env_kwargs"]["controller_configs"]["control_delta"]),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret

    def sample_bc(self, batchsize, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry = self.data[episode_idx][step_idx]
            for k, v in entry.items():
                samples[k].append(v)

        return self._convert_to_batch(samples, device)

    def _stack_actions(self, idx, begin, action_len):
        """stack actions in [begin, end)"""
        episode_idx, step_idx = self.idx2entry[idx]
        episode = self.data[episode_idx]

        actions = []
        valid_actions = []
        for action_idx in range(begin, begin + action_len):
            if action_idx < 0:
                actions.append(torch.zeros_like(episode[step_idx]["action"]))
                valid_actions.append(0)
            elif action_idx < len(episode):
                actions.append(episode[action_idx]["action"])
                valid_actions.append(1)
            else:
                actions.append(torch.zeros_like(actions[-1]))
                valid_actions.append(0)

        valid_actions = torch.tensor(valid_actions, dtype=torch.float32)
        actions = torch.stack(actions, dim=0)
        return actions, valid_actions

    def sample_arnn(self, batchsize, action_cond_horizon, action_pred_horizon, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry = self.data[episode_idx][step_idx]

            if action_cond_horizon:
                cond_actions, _ = self._stack_actions(
                    idx, step_idx - action_cond_horizon, action_cond_horizon
                )
                samples["cond_action"].append(cond_actions)

            actions, valid_actions = self._stack_actions(idx, step_idx - 1, action_pred_horizon + 1)
            input_actions = actions[:-1]
            input_actions[0] = 0
            target_actions = actions[1:]
            assert torch.equal(target_actions[0], entry["action"])

            samples["input_action"].append(input_actions)
            samples["valid_target"].append(valid_actions[1:])
            for k, v in entry.items():
                if k == "action":
                    samples[k].append(target_actions)
                else:
                    samples[k].append(v)

        return self._convert_to_batch(samples, device)

class RobomimicDataloader():
    def __init__(self, dataset, train, context_length=0, step_size=1):
        self.data = []
        obs_key = dataset.cfg.rl_camera
        for d in dataset:
            for i in range(step_size):
                d = d[i:len(d):step_size]
                pad = [d[0] for _ in range(context_length + 1)]
                d = pad + d

                for i in range(context_length, len(d)):
                    if context_length > 0:
                        context = [np.expand_dims(d[j][obs_key].float().numpy()/ 255.0, 0)
                            for j in range(i - context_length, i + 1)]
                        self.data.append(np.expand_dims(np.concatenate(context), 0))
                    else:
                        step = d[i]
                        self.data.append(np.expand_dims(step[obs_key].float().numpy()
                            / 255.0, 0))
                
        self.data = np.concatenate(self.data)
        size = len(dataset.idx2entry)
        if train:
            self.data = self.data[:int(0.95 * float(size))]
        else:
            self.data = self.data[int(0.05 * float(size)):]

    def __getitem__(self, index):
        img = self.data[index]
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    cfg = DatasetConfig(path='/iris/u/jyang27/dev/vqvae/data/square/processed_data96.hdf5')
    dataset = RobomimicDataset(cfg)
    dataloader = RobomimicDataloader(dataset, True, 1)

