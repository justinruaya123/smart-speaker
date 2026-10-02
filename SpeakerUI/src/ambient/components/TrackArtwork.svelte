<script>
  import Icon from './Icon.svelte';
  let { url, compact = false } = $props();
  let failedUrl = $state(null);
  let available = $derived(Boolean(url && url !== failedUrl));
</script>

<span class={compact ? 'album-icon' : 'record'} class:has-artwork={available} aria-hidden="true">
  {#if available}
    {#key url}
      <img src={url} alt="" decoding="async" referrerpolicy="no-referrer" onerror={event => { failedUrl = event.currentTarget.getAttribute('src'); }} />
    {/key}
  {:else if compact}
    <span class="mini-equalizer"><i></i><i></i><i></i><i></i><i></i></span>
  {:else}
    <Icon name="disc-3" />
  {/if}
</span>
