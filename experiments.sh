#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular base name for the weights and
# predictions files. Each single repetition checks if it was already run or is
# currently being run, creates a lockfile, trains the network, computes the
# predictions, and removes the lockfile. To distribute runs between multiple
# GPUs, just run this script multiple times with different --device=cuda:N or
# with different CUDA_VISIBLE_DEVICES=N.
#
# Additional arguments are passed on to each experiment. You may want to consider:
# --device=cuda --cudnn-benchmark --pin-mem --num-workers=4
# And you may want to set a directory for the datasets to live in:
# --data-path=...

here="${0%/*}"
outdir="$here/outputs"

train_if_free() {
	modelname="$1"
	printf "$modelname "
	modeldir="$outdir/models/$modelname"
	mkdir -p "$modeldir"
	if [ ! -f "$modeldir/net_last_model.pth" ]; then
	    if [ ! -f "$modeldir/lock" ]; then
			echo "$HOSTNAME" > "$modeldir/lock"
			echo "starting"
		    python3 "$here"/main.py --output-dir="$outdir" --model-name="$modelname" --save_every=1 --resume="$modeldir/net_checkpoint.pth" "${@:2}"
			rm "$modeldir/lock"
		else
			echo "running"
		fi
	else
	    echo "finished"
	fi
}

train() {
	repeats="$1"
	name="$2"
	for (( r=1; r<=$repeats; r++ )); do
		train_if_free "$name"_r$r --seed=$((r-1)) "${@:3}"
	done
}

bench_if_free() {
	modelname="$1"
	echo "$modelname"
	modeldir="$outdir/models/$modelname"
	mkdir -p "$modeldir"
	if [ ! -f "$modeldir/benchmark_time.txt" ]; then
		if [ ! -f "$modeldir/lock" ]; then
			echo "$HOSTNAME" > "$modeldir/lock"
			python3 "$here"/main.py --output-dir="$outdir" --model-name="$modelname" --frontend-benchmark --benchmark-runs=500 "${@:2}"
			rm "$modeldir/lock"
		fi
	fi
}

# original LEAF
train 3 speechcommands/leaf-original --data-set=SPEECHCOMMANDS "$@"
train 3 crema-d/leaf-original --data-set=CREMAD "$@"
train 3 voxforge/leaf-original --data-set=VOXFORGE "$@"
train 3 nsynth-p/leaf-original --data-set=NSYNTH_PITCH "$@"
train 3 nsynth-i/leaf-original --data-set=NSYNTH_INST "$@"
train 3 birdclef/leaf-original --data-set=BIRDCLEF2021 "$@"
# longer input sizes: requires PCEN parameters to be learned in log space, crashes with NaN otherwise
train 3 birdclef/leaf-original-inputlen8 --data-set=BIRDCLEF2021 --batch-size=32 --input-size=$((16000*8)) --pcen-learn-logs "$@"
train 3 birdclef/leaf-original-inputlen16 --data-set=BIRDCLEF2021 --batch-size=16 --input-size=$((16000*16)) --pcen-learn-logs "$@"
# throughput
bench_if_free benchmark/leaf-original --data-set=SPEECHCOMMANDS "$@"
bench_if_free benchmark/leaf-original-inputlen8 --data-set=BIRDCLEF2021 --batch-size=32 --input-size=$((16000*8)) "$@"
bench_if_free benchmark/leaf-original-inputlen16 --data-set=BIRDCLEF2021 --batch-size=16 --input-size=$((16000*16)) "$@"

# EfficientLeaf + PCEN
model="--frontend=EfficientLeaf --num-groups=4 --conv-win-factor=4.75"
train 3 speechcommands/eleaf4-wf475 --data-set=SPEECHCOMMANDS $model "$@"
train 3 crema-d/eleaf4-wf475 --data-set=CREMAD $model "$@"
train 3 voxforge/eleaf4-wf475 --data-set=VOXFORGE $model "$@"
train 3 nsynth-p/eleaf4-wf475 --data-set=NSYNTH_PITCH $model "$@"
train 3 nsynth-i/eleaf4-wf475 --data-set=NSYNTH_INST $model "$@"
train 3 birdclef/eleaf4-wf475 --data-set=BIRDCLEF2021 $model "$@"
# throughput
bench_if_free benchmark/eleaf4-wf475 --data-set=SPEECHCOMMANDS $model "$@"

# EfficientLeaf + TBN (per band) + appended median filter
model="--frontend=EfficientLeaf --num-groups=4 --conv-win-factor=4.75 --compression=TBN --log1p-initial-a=5 --log1p-trainable --log1p-per-band --tbn-median-filter --tbn-median-filter-append"
train 3 speechcommands/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS $model "$@"
train 3 crema-d/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=CREMAD $model "$@"
train 3 voxforge/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=VOXFORGE $model "$@"
train 3 nsynth-p/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=NSYNTH_PITCH $model "$@"
train 3 nsynth-i/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=NSYNTH_INST $model "$@"
train 3 birdclef/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=BIRDCLEF2021 $model "$@"
train 3 birdclef/eleaf4-wf475-tbn-a5perband-medfiltapp-inputlen8 --data-set=BIRDCLEF2021 --batch-size=32 --input-size=$((16000*8)) $model "$@"
train 3 birdclef/eleaf4-wf475-tbn-a5perband-medfiltapp-inputlen16 --data-set=BIRDCLEF2021 --batch-size=16 --input-size=$((16000*16)) $model "$@"
# throughput
bench_if_free benchmark/eleaf4-wf475-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS $model "$@"
bench_if_free benchmark/eleaf4-wf475-tbn-a5perband-medfiltapp-inputlen8 --data-set=BIRDCLEF2021 --scheduler $model --batch-size=32 --input-size=$((16000*8)) "$@"
bench_if_free benchmark/eleaf4-wf475-tbn-a5perband-medfiltapp-inputlen16 --data-set=BIRDCLEF2021 --scheduler $model --batch-size=16 --input-size=$((16000*16)) "$@"

# Mel + fixed TBN + appended median filter
model="--frontend=Mel --compression=TBN --log1p-initial-a=5 --tbn-median-filter --tbn-median-filter-append"
train 3 speechcommands/mel-tbn-a5fixed-medfiltapp --data-set=SPEECHCOMMANDS $model "$@"
train 3 crema-d/mel-tbn-a5fixed-medfiltapp --data-set=CREMAD $model "$@"
train 3 voxforge/mel-tbn-a5fixed-medfiltapp --data-set=VOXFORGE $model "$@"
train 3 nsynth-p/mel-tbn-a5fixed-medfiltapp --data-set=NSYNTH_PITCH $model "$@"
train 3 nsynth-i/mel-tbn-a5fixed-medfiltapp --data-set=NSYNTH_INST $model "$@"
train 3 birdclef/mel-tbn-a5fixed-medfiltapp --data-set=BIRDCLEF2021 $model "$@"
train 3 birdclef/mel-tbn-a5fixed-medfiltapp-inputlen8 --data-set=BIRDCLEF2021 --batch-size=32 --input-size=$((16000*8)) $model "$@"
train 3 birdclef/mel-tbn-a5fixed-medfiltapp-inputlen16 --data-set=BIRDCLEF2021 --batch-size=16 --input-size=$((16000*16)) $model "$@"
# throughput
bench_if_free benchmark/mel-tbn-a5fixed-medfiltapp --data-set=SPEECHCOMMANDS $model "$@"
bench_if_free benchmark/mel-tbn-a5fixed-medfiltapp-inputlen8 --data-set=BIRDCLEF2021 $model --batch-size=32 --input-size=$((16000*8)) "$@"
bench_if_free benchmark/mel-tbn-a5fixed-medfiltapp-inputlen16 --data-set=BIRDCLEF2021 $model --batch-size=16 --input-size=$((16000*16)) "$@"

# EfficientLeaf + TBN (per band) + appended median filter: hyperparameter optimization
# first measuring throughput
bench_if_free benchmark/eleaf4-wf475 --data-set=SPEECHCOMMANDS --frontend=EfficientLeaf --num-groups=4 --conv-win-factor=4.75 "$@"
for groups in 2 4 8 10; do
    for conv_factor in 6 4.75 3; do
        bench_if_free benchmark/eleaf$groups-wf${conv_factor/./}-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS --scheduler --frontend=EfficientLeaf --num-groups=$groups --conv-win-factor=$conv_factor --compression=TBN --log1p-initial-a=5 --log1p-trainable --log1p-per-band --tbn-median-filter --tbn-median-filter-append --benchmark-runs=100 "$@"
        for stride_factor in 2 3 8 16; do
            bench_if_free benchmark/eleaf$groups-wf${conv_factor/./}-sf${stride_factor}-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS --scheduler --frontend=EfficientLeaf --num-groups=$groups --conv-win-factor=$conv_factor --stride-factor=$stride_factor --compression=TBN --log1p-initial-a=5 --log1p-trainable --log1p-per-band --tbn-median-filter --tbn-median-filter-append --benchmark-runs=100 "$@"
        done
    done
done
# then measuring classification accuracy, for --num-groups=8 (which was fastest overall)
for reps in 1 2 3; do
    for conv_factor in 6 4.75 3; do
        model="--frontend=EfficientLeaf --num-groups=8 --conv-win-factor=$conv_factor --compression=TBN --log1p-initial-a=5 --log1p-trainable --log1p-per-band --tbn-median-filter --tbn-median-filter-append"
        train $reps speechcommands/eleaf8-wf${conv_factor/./}-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS $model "$@"
        for stride_factor in 2 3; do
            train $reps speechcommands/eleaf8-wf${conv_factor/./}-sf${stride_factor}-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS $model --stride-factor=$stride_factor "$@"
        done
        if [ "$conv_factor" == "6" ]; then
            for stride_factor in 8 16; do
                train $reps speechcommands/eleaf8-wf${conv_factor/./}-sf${stride_factor}-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS $model --stride-factor=$stride_factor "$@"
            done
        fi
    done
done

# EfficientLeaf win 6 stride 16 + TBN + appended median filter (no clamp, per band)
model="--frontend=EfficientLeaf --num-groups=8 --conv-win-factor=6 --stride-factor=16 --compression=TBN --log1p-initial-a=5 --log1p-trainable --log1p-per-band --tbn-median-filter --tbn-median-filter-append"
train 3 speechcommands/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp --data-set=SPEECHCOMMANDS $model "$@"
train 3 crema-d/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp --data-set=CREMAD $model "$@"
train 3 voxforge/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp --data-set=VOXFORGE $model "$@"
train 3 nsynth-p/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp --data-set=NSYNTH_PITCH $model "$@"
train 3 nsynth-i/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp --data-set=NSYNTH_INST $model "$@"
train 3 birdclef/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp --data-set=BIRDCLEF2021 $model "$@"
train 3 birdclef/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp-inputlen8 --data-set=BIRDCLEF2021 --batch-size=32 --input-size=$((16000*8)) $model "$@"
train 3 birdclef/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp-inputlen16 --data-set=BIRDCLEF2021 --batch-size=16 --input-size=$((16000*16)) $model "$@"
# throughput with longer input size
bench_if_free benchmark/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp-inputlen8 --data-set=BIRDCLEF2021 --scheduler $model --batch-size=32 --input-size=$((16000*8)) --benchmark-runs=100 "$@"
bench_if_free benchmark/eleaf8-wf6-sf16-tbn-a5perband-medfiltapp-inputlen16 --data-set=BIRDCLEF2021 --scheduler $model --batch-size=16 --input-size=$((16000*16)) --benchmark-runs=100 "$@"
