<script>
  import { icon } from '../icons.js';
  let { name } = $props();
</script>
<!-- Trusted build-time Lucide subset only; icon() rejects names outside the fixed dictionary. No user HTML. -->
{@html icon(name)}
