<script>
  import Icon from './Icon.svelte';
  let { screen, home = false } = $props();
  let m = $derived(screen.model);
  let entries = $derived(m.reminders.map((r, i) => ({r, i})).filter(({r}) => !r.done && (!home || m.remindersExpanded || r.pinned)));
</script>
{#each entries as {r,i}, position (r.id)}
  <div class="reminder-widget" class:is-pinned={r.pinned} class:quiet-row={home && !m.remindersExpanded} data-widget-id={r.id} data-reminder-swipe={i}>
    <button class="reminder-done-action" data-complete-reminder={i} aria-label={`Mark ${r.text} as done`}>Mark as done <Icon name="chevron-right" /></button>
    <div class="reminder-front">
      {#if home && !m.remindersExpanded}
        <button class="quiet-reminder" data-reminder={i} aria-label={r.text}><span><strong>{r.text}</strong>{#if r.deadline}<small>{screen.clockTime(r.deadline)}</small>{/if}</span><span class="reminder-number" aria-hidden="true">{position+1}</span></button>
      {:else}
        <button class="reminder-grip" data-reorder-reminder={i} data-reminder-id={r.id} aria-label={`Reorder ${r.text}; use up and down arrows`}><Icon name="grip-vertical" /></button>
        <button class="reminder-copy" data-reminder={i}><strong>{r.text}</strong>{#if r.deadline}<small>{screen.clockTime(r.deadline)}</small>{/if}</button>
        {#if r.unread}<i class="unread" aria-label="New reminder"></i>{/if}
        <button class="pin-reminder" data-pin-reminder={i} aria-label={`${r.pinned?'Unpin':'Pin'} ${r.text}`} aria-pressed={!!r.pinned}><Icon name="pin" /></button>
      {/if}
    </div>
  </div>
{/each}
{#if entries.length===0 && !home}<p class="empty-reminders">All done</p>{/if}
