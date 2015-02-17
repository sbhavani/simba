% if upload:
    % if error:
    <div class="error">
        <h2>Uploading failed</h2>
        <p>{{error}}.</p>
    </div>
    % else:
    <div class="success">
        <h2>Uploading successful!</h2>
        <p>
	        <em>{{lion}}</em> was successfully uploaded.
	    </p>
    </div>
    % end
% end

<form action=""
      method="POST"
      enctype="multipart/form-data">
    <div>
	<input type="file"
	       name="file" />
        <input type="submit"
	       name="upload"
	       value="Upload" />
    </div>
</form>