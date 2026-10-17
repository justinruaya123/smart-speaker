<script>
  import Iris from './ambient/Iris.svelte';
  const diagnostics = new URLSearchParams(location.search).has('diagnostics');
  const diagnosticPage = diagnostics ? import('./diagnostics/Diagnostics.svelte') : null;
</script>

{#if diagnosticPage}
  {#await diagnosticPage then page}<page.default />{/await}
{:else}
  <Iris />
{/if}
