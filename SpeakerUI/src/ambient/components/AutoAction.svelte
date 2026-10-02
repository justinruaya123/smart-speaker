<script>let {label,action,active=false}=$props();</script>
<button type="button" class="button-primary auto-action" data-action={action}><svg hidden={!active?true:undefined} class="count-stroke" viewBox="0 0 220 62" preserveAspectRatio="none" aria-hidden="true"><rect x="1" y="1" width="218" height="60" rx="30" pathLength="100"></rect></svg>{label}</button>
