(function() {
  
'use strict';
  
  
//フォームのボタンがクリックされたら、またはエンターキーが押されたら
document.search.btn.addEventListener('click', function(e) {
  e.preventDefault();  //画面更新をキャンセル
  
  
  fetch( createURL(document.search.key.value) )
  .then( function( data ) {
    return data.json();  //JSONデータとして取得する
  })
  .then( function( json ) {
    
    createImage( json );
    
  })
})

  
  
//リクエスト用のURLを生成する
function createURL( value ) {
  var API_KEY = '21400938-689d2bd4da919c15b4230d8e5';
  var baseUrl = 'https://pixabay.com/api/?key=' + API_KEY;
  var keyword = '&q=' + encodeURIComponent( value );
  var option = '&orientation=horizontal&per_page=100';
  var URL = baseUrl + keyword + option;
  
  return URL;
}
 
  
//画像のJSONデータを画面に表示する
function createImage( json ) {
  var result = document.getElementById('result');

  result.innerHTML = '';  //検索するごとに画面をリセットする

  //該当する画像があれば
  if( json.totalHits > 0 ) {
    json.hits.forEach( function( value ) {
      var img = document.createElement('img');
      var a = document.createElement('a');

      // a.href = value.pageURL;  //ダウンロード用のサイトURL
      a.target = '_blank';
      img.src = value.previewURL;  //プレビュー用の画像URL
      
      a.appendChild( img );
      result.appendChild( a );

      // 検索画像をクリックした時
      a.addEventListener('click', function(e) {
        img_obj.src = value.largeImageURL;  //画像URL
        img_url.value = value.largeImageURL;  //画像URL
      })
      
    })
  }
  else {
    alert('該当する写真がありません');
  }
}

// ------------------------------------------------------------
// model load
// ------------------------------------------------------------
document.load.btn.addEventListener('click', function(e) {
  // e.preventDefault();  //画面更新をキャンセル
  // カウントアップ処理
  var count = 0;
  var timeload = null;

  // timeload = setInterval(loaddown, 1000);

  setInterval(function(){
    //code goes here that will be run every 5 seconds.
    document.getElementById('time_load').value = count;
    // alert('count更新後');

    if (count >= 180) {
      clearInterval(timeload);
    }

    count++;
    // alert(count);

  }, 500);

})

})();