accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=/fastdata/sd-models/sd21-unclip-h.ckpt
    --dataset_config=./datasets/fler.toml 
    --output_dir=/fastdata/lora-outputs 
    --output_name=fler_train 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=400 
    --learning_rate=1e-4 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora

is_dreambooth: true
is_controlnet: false
subset_params_klass = DreamBoothSubsetParams
dataset_params_klass = DreamBoothDatasetParams

@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
    is_reg: bool = False
    class_tokens: Optional[str] = None
    caption_extension: str = ".caption"
    cache_info: bool = False
    alpha_mask: bool = False

class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        cache_info: bool,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator: str,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir

def generate_dataset_group_by_blueprint(dataset_group_blueprint: DatasetGroupBlueprint):
    datasets: List[Union[DreamBoothDataset, FineTuningDataset, ControlNetDataset]] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        if dataset_blueprint.is_controlnet:
            subset_klass = ControlNetSubset
            dataset_klass = ControlNetDataset
        elif dataset_blueprint.is_dreambooth:
            subset_klass = DreamBoothSubset
            dataset_klass = DreamBoothDataset
        else:
            subset_klass = FineTuningSubset
            dataset_klass = FineTuningDataset

        subsets = [subset_klass(**asdict(subset_blueprint.params)) for subset_blueprint in dataset_blueprint.subsets]
        dataset = dataset_klass(subsets=subsets, **asdict(dataset_blueprint.params))
        datasets.append(dataset)

    # print info
    info = ""
    for i, dataset in enumerate(datasets):
        is_dreambooth = isinstance(dataset, DreamBoothDataset)
        is_controlnet = isinstance(dataset, ControlNetDataset)
        info += dedent(
            f"""\
      [Dataset {i}]
        batch_size: {dataset.batch_size}
        resolution: {(dataset.width, dataset.height)}
        enable_bucket: {dataset.enable_bucket}
        network_multiplier: {dataset.network_multiplier}
    """
        )

        if dataset.enable_bucket:
            info += indent(
                dedent(
                    f"""\
        min_bucket_reso: {dataset.min_bucket_reso}
        max_bucket_reso: {dataset.max_bucket_reso}
        bucket_reso_steps: {dataset.bucket_reso_steps}
        bucket_no_upscale: {dataset.bucket_no_upscale}
      \n"""
                ),
                "  ",
            )
        else:
            info += "\n"

        for j, subset in enumerate(dataset.subsets):
            info += indent(
                dedent(
                    f"""\
        [Subset {j} of Dataset {i}]
          image_dir: "{subset.image_dir}"
          image_count: {subset.img_count}
          num_repeats: {subset.num_repeats}
          shuffle_caption: {subset.shuffle_caption}
          keep_tokens: {subset.keep_tokens}
          keep_tokens_separator: {subset.keep_tokens_separator}
          caption_separator: {subset.caption_separator}
          secondary_separator: {subset.secondary_separator}
          enable_wildcard: {subset.enable_wildcard}
          caption_dropout_rate: {subset.caption_dropout_rate}
          caption_dropout_every_n_epoches: {subset.caption_dropout_every_n_epochs}
          caption_tag_dropout_rate: {subset.caption_tag_dropout_rate}
          caption_prefix: {subset.caption_prefix}
          caption_suffix: {subset.caption_suffix}
          color_aug: {subset.color_aug}
          flip_aug: {subset.flip_aug}
          face_crop_aug_range: {subset.face_crop_aug_range}
          random_crop: {subset.random_crop}
          token_warmup_min: {subset.token_warmup_min},
          token_warmup_step: {subset.token_warmup_step},
          alpha_mask: {subset.alpha_mask},
      """
                ),
                "  ",
            )

            if is_dreambooth:
                info += indent(
                    dedent(
                        f"""\
          is_reg: {subset.is_reg}
          class_tokens: {subset.class_tokens}
          caption_extension: {subset.caption_extension}
        \n"""
                    ),
                    "    ",
                )
            elif not is_controlnet:
                info += indent(
                    dedent(
                        f"""\
          metadata_file: {subset.metadata_file}
        \n"""
                    ),
                    "    ",
                )

    logger.info(f"{info}")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no
    for i, dataset in enumerate(datasets):
        logger.info(f"[Dataset {i}]")
        dataset.make_buckets()
        dataset.set_seed(seed)

    return DatasetGroup(datasets)




subset_klass = DreamBoothSubset
dataset_klass = DreamBoothDataset

# Dataset in use:

DreamBoothDataset

# Subset in use:

DreamBoothSubset

# Subset params in use:

