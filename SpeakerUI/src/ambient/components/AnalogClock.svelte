<script>
  let { time, className = '' } = $props();
  let parts = $derived(time.split(':').map(Number));
</script>
<span class="analog-clock {className}" role="img" aria-label={time} style:--hour-angle={`${(parts[0] % 12) * 30 + parts[1] * .5}deg`} style:--minute-angle={`${parts[1] * 6}deg`}><i class="hour-hand"></i><i class="minute-hand"></i><i class="clock-pivot"></i></span>
